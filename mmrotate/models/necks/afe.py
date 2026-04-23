# mmrotate/models/necks/afe.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List


# ========== 辅助函数：批量估计角度（与 FAA 一致） ==========
def estimate_main_direction_batch(patch_tensor, eps=1e-8):
    """
    批量估计每个 patch 的主轴方向 (输入: [B*N, 1, m, m])
    返回: [B*N] 弧度 (0~pi)
    """
    Bn, _, m, _ = patch_tensor.shape
    device = patch_tensor.device

    x_fft = torch.fft.fft2(patch_tensor.squeeze(1), norm='ortho')
    x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
    mag = x_fft_shifted.abs() + eps

    # 频率网格
    h_freq = torch.fft.fftfreq(m) * m
    w_freq = torch.fft.fftfreq(m) * m
    h_grid, w_grid = torch.meshgrid(h_freq, w_freq, indexing='ij')
    h_grid = torch.fft.fftshift(h_grid).to(device)
    w_grid = torch.fft.fftshift(w_grid).to(device)

    rho = torch.sqrt(h_grid ** 2 + w_grid ** 2)
    theta = torch.atan2(h_grid, w_grid)
    theta = (theta + 2 * math.pi) % (2 * math.pi)

    mask = rho > eps
    rho_valid = rho[mask]
    theta_valid = theta[mask]

    mag_flat = mag.view(Bn, -1)
    mag_valid = mag_flat[:, mask.view(-1)]
    weighted_energy = mag_valid * rho_valid.unsqueeze(0)
    max_idx = torch.argmax(weighted_energy, dim=1)
    theta_e = theta_valid[max_idx]
    return theta_e % math.pi


def compute_angle_map(x, window_size=7):
    """
    输入特征图 x: [B, C, H, W]
    输出角度图 angle_map: [B, H, W] 弧度 (0~pi)
    通过滑动窗口 (stride=1, padding=window_size//2) 对每个位置估计方向
    """
    B, C, H, W = x.shape
    # 通道取均值以降低计算量
    x_mean = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
    # 使用 unfold 提取所有窗口，步长=1，边缘补零 (padding)
    pad = window_size // 2
    x_pad = F.pad(x_mean, (pad, pad, pad, pad), mode='reflect')
    patches = F.unfold(x_pad, kernel_size=window_size, stride=1)  # [B, m*m, H*W]
    N = patches.shape[2]
    patches = patches.transpose(1, 2).reshape(B * N, 1, window_size, window_size)
    angles = estimate_main_direction_batch(patches)  # [B*N]
    angle_map = angles.view(B, H, W)
    return angle_map


# ========== 动态方向卷积模块 ==========
class DynamicDirectionalConv(nn.Module):
    """
    动态方向深度卷积：根据每个位置的预测角度（或权重）线性组合基础核，对输入特征进行 depthwise 卷积。
    每个空间位置独立生成核（参数为 H×W×4 个标量）
    """
    def __init__(self, in_channels, kernel_size=7, stride=1, padding=3):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 预定义4个基础核（0°,45°,90°,135°方向的各向异性高斯核）
        base_kernels = self._create_base_kernels(kernel_size)  # [4, 1, K, K]
        self.register_buffer('base_kernels', base_kernels)

        # 权重预测网络：输入角度（sin2θ, cos2θ），输出4个组合权重
        self.weight_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Softmax(dim=-1)   # 权重和为1，保证数值稳定
        )

    def _create_base_kernels(self, k):
        """生成4个方向的各向异性高斯核（长轴σ1=2.5，短轴σ2=1.0）"""
        sigma1, sigma2 = 2.5, 1.0
        angles = [0, math.pi/4, math.pi/2, 3*math.pi/4]
        kernels = []
        for theta in angles:
            kernel = self._gaussian_kernel(k, sigma1, sigma2, theta)
            kernels.append(kernel)
        return torch.stack(kernels, dim=0).unsqueeze(1)  # [4,1,k,k]

    def _gaussian_kernel(self, size, sigma1, sigma2, theta):
        """生成旋转的2D高斯核，长轴sigma1，短轴sigma2，方向theta"""
        ax = torch.linspace(-(size//2), size//2, size)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        x_rot = xx * math.cos(theta) + yy * math.sin(theta)
        y_rot = -xx * math.sin(theta) + yy * math.cos(theta)
        kernel = torch.exp(-(x_rot**2 / (2*sigma1**2) + y_rot**2 / (2*sigma2**2)))
        kernel = kernel / kernel.sum()   # 归一化，使和为1
        return kernel

    def forward(self, x, angle_map):
        B, C, H, W = x.shape
        device = x.device

        sin2 = torch.sin(2 * angle_map)
        cos2 = torch.cos(2 * angle_map)
        angle_feat = torch.stack([sin2, cos2], dim=-1)  # [B,H,W,2]

        weights = self.weight_net(angle_feat)  # [B,H,W,4]

        base = self.base_kernels  # [4, 1, K, K]
        # 修正 einsum：用 'kchw' 代替 'kckl'，避免重复下标
        combined = torch.einsum('bhwk,kchw->bhwchw', weights, base)  # [B,H,W,1,K,K]
        combined = combined.squeeze(3)  # [B,H,W,K,K]

        x_pad = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        patches = F.unfold(x_pad, kernel_size=self.kernel_size, stride=self.stride)  # [B, C*K*K, N]
        N = patches.shape[-1]
        patches = patches.view(B, C, self.kernel_size * self.kernel_size, N)  # [B,C,49,N]
        patches = patches.permute(0, 1, 3, 2)  # [B,C,N,49]

        kernels_flat = combined.view(B, H * W, -1)  # [B, N, 49]

        out = torch.einsum('bcnk,bnk->bcn', patches, kernels_flat)  # [B,C,N]
        out = out.view(B, C, H, W)
        return out


# ========== 集成到 FPN 的 Neck ==========
@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """
    使用动态方向卷积增强 FPN 的自上而下融合。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes: List[str],
                 **kwargs):
        super(AngleFreqEnhanceFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        self.fusion_modes = fusion_modes
        # 动态卷积模块列表
        self.dynamic_convs = nn.ModuleList()
        for mode in fusion_modes:
            if mode == 'afe':
                self.dynamic_convs.append(DynamicDirectionalConv(out_channels))
            else:
                self.dynamic_convs.append(None)

    @auto_fp16()
    def forward(self, inputs):
        """inputs: backbone 输出 (list of tensor)"""
        # 1. 横向连接
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        # 2. 自上而下融合
        for i in range(used_backbone_levels - 1, 0, -1):
            fusion_idx = used_backbone_levels - 1 - i   # 0,1,2...
            mode = self.fusion_modes[fusion_idx]

            if mode == 'afe':
                # 对低层特征 laterals[i-1] 应用动态方向卷积
                angle_map = compute_angle_map(laterals[i-1])   # 计算角度图
                enhanced_low = self.dynamic_convs[fusion_idx](laterals[i-1], angle_map)
                # 上采样高层特征并相加
                up_high = F.interpolate(laterals[i], size=enhanced_low.shape[-2:], **self.upsample_cfg)
                laterals[i-1] = enhanced_low + up_high
            else:   # mode == 'add'
                up_high = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], **self.upsample_cfg)
                laterals[i-1] = laterals[i-1] + up_high

        # 3. 输出卷积
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # 4. 生成额外输出层 (与原 FPN 相同)
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)