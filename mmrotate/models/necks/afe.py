import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import ROTATED_NECKS

@ROTATED_NECKS.register_module()
class AFEModule(nn.Module):
    """
    AngleFreqEnhance (AFE) Module - 方案二实现
    基于FFT幅度谱生成各向异性高斯卷积核，增强旋转目标特征。
    """

    def __init__(self, channels, m=7, eps=1e-8):
        super(AFEModule, self).__init__()
        self.m = m
        self.channels = channels
        self.eps = eps

        # 提取参数 theta, lambda1, lambda2 的轻量级网络
        # 输入是 m*m 的幅度谱，输出 3 个几何参数
        self.param_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(m * m, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 输出: sin_2theta, cos_2theta, lambda_ratio (或直接回归参数)
        )

        # 实部与虚部共享的 1x1 整合卷积
        self.refine_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def generate_anisotropic_kernel(self, params, device):
        """
        方案二核心：根据主轴参数生成 m*m 的各向异性高斯核
        """
        B = params.shape[0]
        # 解码参数 (为了保证连续性，建议预测 sin2theta 和 cos2theta)
        theta = torch.atan2(params[:, 0], params[:, 1]) / 2.0
        # 修正角度：物体主方向与频域主轴垂直
        theta = theta + (math.pi / 2.0)

        lam1 = torch.exp(params[:, 2])  # 长度参数，使用exp保证为正
        lam2 = 1.0 / (lam1 + self.eps)  # 另一轴反向缩放，保持各向异性

        # 创建 7x7 坐标网格
        y, x = torch.meshgrid(
            torch.linspace(-(self.m // 2), self.m // 2, self.m, device=device),
            torch.linspace(-(self.m // 2), self.m // 2, self.m, device=device)
        )
        x = x.repeat(B, 1, 1)  # [B, m, m]
        y = y.repeat(B, 1, 1)

        # 旋转坐标系
        cos_t, sin_t = torch.cos(theta).view(B, 1, 1), torch.sin(theta).view(B, 1, 1)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t

        # 计算高斯分布 (各向异性)
        # lam1, lam2 控制核在不同方向的拉伸
        kernel = torch.exp(
            -((x_rot ** 2) / (2 * lam1.view(B, 1, 1) ** 2) + (y_rot ** 2) / (2 * lam2.view(B, 1, 1) ** 2)))

        # 归一化
        kernel_sum = kernel.sum(dim=(1, 2), keepdim=True) + self.eps
        kernel = kernel / kernel_sum
        return kernel  # [B, m, m]

    def forward(self, x):
        B, C, H, W = x.shape
        m = self.m

        # 1. FFT 中心化并分离实部虚部
        # 注意：对于C2/C3等高分辨率层，建议分window处理，类似FAA
        x_f = torch.fft.fft2(x, norm='ortho')
        x_f = torch.fft.fftshift(x_f, dim=(-2, -1))

        real = x_f.real  # [B, C, H, W]
        imag = x_f.imag  # [B, C, H, W]

        # 2. 提取幅度谱主轴参数
        # 方案：对幅度谱进行全局池化得到一个代表性的 m*m 谱 (或采样中心区域)
        mag = torch.abs(x_f)
        mag_weighted = torch.mean(mag, dim=1, keepdim=True)  # 压缩通道至1 [B, 1, H, W]

        # 截取中心 m*m 区域获取主方向信息
        h_idx, w_idx = H // 2, W // 2
        mag_center = mag_weighted[:, :, h_idx - m // 2: h_idx + m // 2 + 1, w_idx - m // 2: w_idx + m // 2 + 1]

        params = self.param_net(mag_center)  # [B, 3]

        # 3. 生成动态卷积核
        dynamic_kernel = self.generate_anisotropic_kernel(params, x.device)  # [B, m, m]
        # 转换为深度可分离卷积格式 [B*C, 1, m, m]
        dynamic_kernel = dynamic_kernel.repeat_interleave(C, dim=0).unsqueeze(1)

        # 4. 执行深度卷积 (实部与虚部使用相同核)
        # padding 保证尺寸不变
        real_f = F.conv2d(real.view(1, B * C, H, W), dynamic_kernel,
                          padding=m // 2, groups=B * C).view(B, C, H, W)
        imag_f = F.conv2d(imag.view(1, B * C, H, W), dynamic_kernel,
                          padding=m // 2, groups=B * C).view(B, C, H, W)

        # 5. 1x1 卷积信息整合 (共享参数)
        real_f = self.refine_conv(real_f)
        imag_f = self.refine_conv(imag_f)

        # 6. 重组并 iFFT 返回空域
        out_f = torch.complex(real_f, imag_f)
        out_f = torch.fft.ifftshift(out_f, dim=(-2, -1))
        out = torch.fft.ifft2(out_f, norm='ortho').real

        return out

@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(nn.Module):
    """
    AngleFreqEnhance (AFE) Module
    功能：
    1. 接收 ResNet/FPN 的两层输入。
    2. 对低层特征 x_low 进行 FFT 变换，利用其中心幅度谱回归方向参数。
    3. 方案二实现：生成 7x7 各向异性高斯卷积核（动态权重）。
    4. 在频域分离实部与虚部，应用该动态卷积核。
    5. 重组并逆 FFT 变换回空域，最后与高层特征融合。
    """

    def __init__(self, channels, m=7, eps=1e-8):
        super(AngleFreqEnhanceFPN, self).__init__()
        self.m = m
        self.channels = channels
        self.eps = eps

        # 提取参数 theta, lambda1, lambda2 的轻量级网络
        # 连续性保护：回归 sin(2theta), cos(2theta) 以及 log(lambda)
        self.param_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(m * m, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        # 实部与虚部共享的 1x1 信息整合卷积
        self.refine_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # 用于对齐高层特征的通道（如果通道数不一致可开启）
        # self.up_conv = nn.Conv2d(high_channels, channels, kernel_size=1)

    def generate_anisotropic_kernel(self, params, device):
        """
        方案二核心实现：生成具有方向性的各向异性高斯核
        """
        B = params.shape[0]

        # 1. 连续性保护：使用 atan2 处理 sin/cos 预测值
        # params[:, 0] = sin(2theta), params[:, 1] = cos(2theta)
        theta = torch.atan2(params[:, 0], params[:, 1]) / 2.0

        # 2. 角度修正：遥感物体物理方向与频域能量轴垂直，需补偿 90 度
        theta = theta + (math.pi / 2.0)

        # 3. 尺度保护：使用 exp 确保 lambda > 0，预测 log 空间的值
        lam1 = torch.exp(params[:, 2])
        lam2 = 1.0 / (lam1 + self.eps)

        # 4. 生成 7x7 坐标网格
        y, x = torch.meshgrid(
            torch.linspace(-(self.m // 2), self.m // 2, self.m, device=device),
            torch.linspace(-(self.m // 2), self.m // 2, self.m, device=device)
        )
        x = x.repeat(B, 1, 1)  # [B, m, m]
        y = y.repeat(B, 1, 1)

        # 5. 旋转坐标系映射
        cos_t, sin_t = torch.cos(theta).view(B, 1, 1), torch.sin(theta).view(B, 1, 1)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t

        # 6. 计算各向异性高斯分布
        kernel = torch.exp(
            -((x_rot ** 2) / (2 * lam1.view(B, 1, 1) ** 2) + (y_rot ** 2) / (2 * lam2.view(B, 1, 1) ** 2)))

        # 7. 归一化，确保权重能量守恒
        kernel_sum = kernel.sum(dim=(1, 2), keepdim=True) + self.eps
        kernel = kernel / kernel_sum
        return kernel  # [B, m, m]

    def forward(self, x_high, x_low):
        """
        x_high: 来自深层特征 [B, C, H_h, W_h]
        x_low:  来自当前层特征 [B, C, H_l, W_l]
        """
        B, C, H, W = x_low.shape
        m = self.m

        # --- 步骤 1: 频域转换与分离 ---
        x_f = torch.fft.fft2(x_low, norm='ortho')
        x_f = torch.fft.fftshift(x_f, dim=(-2, -1))

        real = x_f.real
        imag = x_f.imag

        # --- 步骤 2: 提取方向先验参数 ---
        # 压缩通道取幅度谱均值，提取最显著的能量方向 [B, 1, H, W]
        mag = torch.abs(x_f)
        mag_weighted = torch.mean(mag, dim=1, keepdim=True)

        # 截取中心 m*m (7x7) 区域，该区域包含全局物体的方向分量
        h_idx, w_idx = H // 2, W // 2
        mag_center = mag_weighted[:, :, h_idx - m // 2: h_idx + m // 2 + 1, w_idx - m // 2: w_idx + m // 2 + 1]

        # 通过参数网回归得到 theta 和 lambda
        params = self.param_net(mag_center)  # [B, 3]

        # --- 步骤 3: 动态卷积核生成与应用 ---
        dynamic_kernel = self.generate_anisotropic_kernel(params, x_low.device)
        # 转换为深度可分离卷积格式 [B*C, 1, m, m]
        dynamic_kernel = dynamic_kernel.repeat_interleave(C, dim=0).unsqueeze(1)

        # 对实部和虚部执行相同的可分离卷积
        # 使用 padding=m//2 (3) 保持空间尺寸不变
        real_f = F.conv2d(real.view(1, B * C, H, W), dynamic_kernel,
                          padding=m // 2, groups=B * C).view(B, C, H, W)
        imag_f = F.conv2d(imag.view(1, B * C, H, W), dynamic_kernel,
                          padding=m // 2, groups=B * C).view(B, C, H, W)

        # --- 步骤 4: 信息整合与逆变换 ---
        # 共享 1x1 卷积权重以保持复数相关性
        real_f = self.refine_conv(real_f)
        imag_f = self.refine_conv(imag_f)

        out_f = torch.complex(real_f, imag_f)
        out_f = torch.fft.ifftshift(out_f, dim=(-2, -1))
        x_low_enhanced = torch.fft.ifft2(out_f, norm='ortho').real

        # --- 步骤 5: 特征融合 (与 FAA 逻辑一致) ---
        # 将深层特征上采样到当前层尺寸并融合
        x_high_upsampled = F.interpolate(x_high, size=(H, W), mode='bilinear', align_corners=False)
        fused = x_low_enhanced + x_high_upsampled

        return fused