import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List

EPS = 1e-8


def estimate_main_direction_batch(patch_tensor, eps=EPS):
    """
    patch_tensor: [B*N, 1, m, m]
    Returns: [B*N] angles in [0, pi)
    """
    Bn, _, m, _ = patch_tensor.shape
    device = patch_tensor.device
    x_fft = torch.fft.fft2(patch_tensor.squeeze(1), norm='ortho')
    x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
    mag = x_fft_shifted.abs() + eps

    h_freq = torch.fft.fftfreq(m, d=1.0) * m
    w_freq = torch.fft.fftfreq(m, d=1.0) * m
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


def compute_angle_map(x, window_size=7, stride=1):
    """
    x: [B, C, H, W]
    Returns: [B, H, W] angle in radians [0, pi)
    """
    B, C, H, W = x.shape
    x_mean = x.mean(dim=1, keepdim=True)
    pad = window_size // 2
    x_pad = F.pad(x_mean, (pad, pad, pad, pad), mode='reflect')
    patches = F.unfold(x_pad, kernel_size=window_size, stride=stride)
    N = patches.shape[-1]
    patches = patches.transpose(1, 2).reshape(B * N, 1, window_size, window_size)
    angles = estimate_main_direction_batch(patches)
    angle_map = angles.view(B, H, W)
    angle_map = torch.nan_to_num(angle_map, nan=0.0, posinf=math.pi, neginf=0.0)
    return angle_map


class AngleAdaptiveConv(nn.Module):
    """
    角度自适应深度卷积（旋转卷积核版本）
    输入输出通道相同（默认 64），使用预旋转 60 个离散角度 + 向量化点积。
    内部计算角度时额外加上 90°（π/2），使卷积核长轴对齐目标长轴。
    """
    def __init__(self, in_channels, kernel_size=7, stride=1, padding=None, num_angles=60):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding
        self.num_angles = num_angles

        # 可学习的基础卷积核（深度卷积形式，每个通道独立）
        self.base_kernel = nn.Parameter(
            self._create_gaussian_kernel(kernel_size, sigma=0.5).repeat(in_channels, 1, 1, 1)
        )
        # 可学习的角度残差（初始 0）
        self.angle_bias = nn.Parameter(torch.tensor(0.0))

        # 预定义离散角度（弧度），范围 [0, π)
        self.register_buffer('angle_grid', torch.linspace(0, math.pi, num_angles, dtype=torch.float32))

    def _create_gaussian_kernel(self, k, sigma):
        ax = torch.linspace(-(k//2), k//2, k)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2*sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # [1,1,k,k]

    def _rotate_kernel(self, kernel, angle):
        """旋转基础核，kernel: [C,1,k,k], angle: scalar Tensor"""
        C, _, k, _ = kernel.shape
        device = kernel.device
        center = (k - 1) / 2.0
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        theta = torch.tensor([
            [cos_a, -sin_a, center - cos_a*center + sin_a*center],
            [sin_a,  cos_a, center - sin_a*center - cos_a*center]
        ], device=device).unsqueeze(0)  # [1,2,3]
        kernel_4d = kernel.view(C, 1, k, k)
        grid = F.affine_grid(theta.expand(C, -1, -1), kernel_4d.size(), align_corners=False)
        rotated = F.grid_sample(kernel_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return rotated  # [C,1,k,k]

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_channels
        k = self.kernel_size
        pad = self.padding

        # 1. 计算角度（FFT主轴方向 + 可学习残差 + 固定偏移 π/2）
        angle_fft = compute_angle_map(x, window_size=k, stride=self.stride)   # [B, H, W]
        # 注：compute_angle_map 内部已经对输入做了通道均值，所以这里输入 x 可以是任意通道数
        angle = angle_fft + self.angle_bias + math.pi / 2   # 加 90°
        angle = angle % math.pi            # 限制到 [0, π)
        angle = torch.clamp(angle, 0, math.pi - 1e-6)

        # 2. 预旋转所有离散角度的核
        pre_rotated = []
        for i in range(self.num_angles):
            theta_i = self.angle_grid[i]
            rot_kernel = self._rotate_kernel(self.base_kernel, theta_i)   # [C,1,k,k]
            pre_rotated.append(rot_kernel.view(C, k*k))
        pre_rotated = torch.stack(pre_rotated, dim=0)   # [num_angles, C, k*k]

        # 3. 获取 patches 并展平
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        patches = F.unfold(x_pad, kernel_size=k, stride=self.stride)   # [B, C*k*k, N]
        N = patches.shape[-1]   # H_out * W_out
        patches = patches.view(B, C, k*k, N).permute(0, 1, 3, 2)        # [B, C, N, k*k]
        patches_flat = patches.permute(0, 2, 1, 3).reshape(B*N, C, k*k) # [B*N, C, k*k]

        # 4. 量化角度
        angle_flat = angle.reshape(-1)   # [B*N]
        indices = torch.argmin(torch.abs(angle_flat.unsqueeze(1) - self.angle_grid.unsqueeze(0)), dim=1)  # [B*N]

        # 5. 向量化深度卷积
        selected_kernels = pre_rotated[indices]          # [B*N, C, k*k]
        out_flat = (patches_flat * selected_kernels).sum(dim=2)   # [B*N, C]

        out = out_flat.view(B, C, H, W)
        return out


class MultiBranchAFE(nn.Module):
    """
    四分支多尺度方向增强模块：
      - 分支1：恒等映射
      - 分支2：1x7 深度可分离卷积（水平）
      - 分支3：7x1 深度可分离卷积（垂直）
      - 分支4：角度自适应卷积（旋转核 + 90° 偏移）
    输入通道假设为 256，被均分为 4 份（每份 64）。
    """
    def __init__(self, in_channels=256, kernel_size=7, stride=1, padding=None):
        super().__init__()
        assert in_channels % 4 == 0, "in_channels must be divisible by 4"
        self.in_channels = in_channels
        self.branch_channels = in_channels // 4   # 64
        self.kernel_size = kernel_size
        self.stride = stride
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

        # 分支2: 1x7 深度可分离卷积
        self.dw_horiz = nn.Conv2d(self.branch_channels, self.branch_channels,
                                  kernel_size=(1, kernel_size), stride=stride,
                                  padding=(0, padding), groups=self.branch_channels, bias=False)
        # 分支3: 7x1 深度可分离卷积
        self.dw_vert = nn.Conv2d(self.branch_channels, self.branch_channels,
                                 kernel_size=(kernel_size, 1), stride=stride,
                                 padding=(padding, 0), groups=self.branch_channels, bias=False)
        # 分支4: 角度自适应卷积
        self.angle_conv = AngleAdaptiveConv(self.branch_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding)

        # 融合卷积（1x1）
        self.fusion = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # 按通道分割
        ch = self.branch_channels
        x1 = x[:, :ch, :, :]          # 恒等
        x2 = x[:, ch:2*ch, :, :]      # 水平
        x3 = x[:, 2*ch:3*ch, :, :]    # 垂直
        x4 = x[:, 3*ch:4*ch, :, :]    # 角度自适应

        # 处理各分支
        out1 = x1
        out2 = self.dw_horiz(x2)
        out3 = self.dw_vert(x3)
        out4 = self.angle_conv(x4)

        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.fusion(out)
        out = self.bn_fusion(out)
        out = self.act(out)
        return out


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """
    增强版 FPN，支持可配置的 AFE 模块参数。
    当 fusion_mode 为 'afe' 时，使用 MultiBranchAFE 进行方向增强融合。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes: List[str],
                 afe_kernel_size=7,        # 仅保留 kernel_size 参数
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        self.fusion_modes = fusion_modes
        self.dynamic_convs = nn.ModuleList()
        for mode in fusion_modes:
            if mode == 'afe':
                # 注意：MultiBranchAFE 内部固定输入输出通道数为 out_channels（必须能被 4 整除）
                assert out_channels % 4 == 0, "out_channels must be divisible by 4 for AFE"
                self.dynamic_convs.append(
                    MultiBranchAFE(
                        in_channels=out_channels,
                        kernel_size=afe_kernel_size
                    )
                )
            else:
                self.dynamic_convs.append(None)

    @auto_fp16()
    def forward(self, inputs):
        # 同原始 FPN 的 lateral 构建
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        # 自顶向下融合
        for i in range(used_backbone_levels - 1, 0, -1):
            fusion_idx = used_backbone_levels - 1 - i
            mode = self.fusion_modes[fusion_idx]

            if mode == 'add':
                if 'scale_factor' in self.upsample_cfg:
                    upsampled = F.interpolate(laterals[i], **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
                laterals[i - 1] = laterals[i - 1] + upsampled

            elif mode == 'afe':
                enhanced_low = self.dynamic_convs[fusion_idx](laterals[i - 1])
                up_high = F.interpolate(laterals[i], size=enhanced_low.shape[-2:], **self.upsample_cfg)
                laterals[i - 1] = enhanced_low + up_high

            else:
                raise ValueError(f"Unknown fusion mode: {mode}")

        # 输出层
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # 生成额外层（同原 FPN）
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