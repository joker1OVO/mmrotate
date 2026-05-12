import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List

EPS = 1e-8


# ========== 角度估计函数（与之前相同）==========
def estimate_main_direction_batch(patch_tensor, eps=EPS):
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


# ========== 旋转自适应卷积模块（简化版，预旋转60角度）==========
class AngleAdaptiveConv(nn.Module):
    """旋转卷积核，基于FFT角度+可学习偏置+90°偏移，预旋转60个离散角度，向量化计算"""
    def __init__(self, in_channels, kernel_size=7, stride=1, padding=None, num_angles=60):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding
        self.num_angles = num_angles

        self.base_kernel = nn.Parameter(
            self._gaussian_kernel(kernel_size, sigma=0.5).repeat(in_channels, 1, 1, 1)
        )
        self.angle_bias = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('angle_grid', torch.linspace(0, math.pi, num_angles, dtype=torch.float32))

    def _gaussian_kernel(self, k, sigma):
        ax = torch.linspace(-(k//2), k//2, k)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2*sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def _rotate_kernel(self, kernel, angle):
        C, _, k, _ = kernel.shape
        device = kernel.device
        center = (k - 1) / 2.0
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        theta = torch.tensor([
            [cos_a, -sin_a, center - cos_a*center + sin_a*center],
            [sin_a,  cos_a, center - sin_a*center - cos_a*center]
        ], device=device).unsqueeze(0)
        kernel_4d = kernel.view(C, 1, k, k)
        grid = F.affine_grid(theta.expand(C, -1, -1), kernel_4d.size(), align_corners=False)
        rotated = F.grid_sample(kernel_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return rotated

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_channels
        k = self.kernel_size
        pad = self.padding

        # 计算角度（FFT主方向 + 可学习偏置 + 90°偏移）
        angle_fft = compute_angle_map(x, window_size=k, stride=self.stride)
        angle = angle_fft + self.angle_bias + math.pi / 2
        angle = angle % math.pi
        angle = torch.clamp(angle, 0, math.pi - 1e-6)

        # 预旋转所有离散角度的核
        pre_rotated = []
        for i in range(self.num_angles):
            theta_i = self.angle_grid[i]
            rot_k = self._rotate_kernel(self.base_kernel, theta_i)  # [C,1,k,k]
            pre_rotated.append(rot_k.view(C, k*k))
        pre_rotated = torch.stack(pre_rotated, dim=0)   # [num_angles, C, k*k]

        # 提取 patches
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        patches = F.unfold(x_pad, kernel_size=k, stride=self.stride)  # [B, C*k*k, N]
        N = patches.shape[-1]
        patches = patches.view(B, C, k*k, N).permute(0, 1, 3, 2)       # [B, C, N, k*k]
        patches_flat = patches.permute(0, 2, 1, 3).reshape(B*N, C, k*k) # [B*N, C, k*k]

        # 量化角度
        angle_flat = angle.reshape(-1)  # [B*N]
        indices = torch.argmin(torch.abs(angle_flat.unsqueeze(1) - self.angle_grid.unsqueeze(0)), dim=1)  # [B*N]

        # 批量卷积（向量化点积）
        selected_kernels = pre_rotated[indices]   # [B*N, C, k*k]
        out_flat = (patches_flat * selected_kernels).sum(dim=2)  # [B*N, C]
        out = out_flat.view(B, C, H, W)
        return out


# ========== 通道注意力（SE）==========
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ========== 四分支主模块 ==========
class MultiBranchAFE(nn.Module):
    """四分支多尺度方向增强模块（即插即用）"""
    def __init__(self,
                 in_channels,
                 kernel_size_small=7,    # 方形卷积核尺寸
                 kernel_size_asym1=7,    # 第一组非对称卷积核尺寸
                 kernel_size_asym2=13,   # 第二组非对称卷积核尺寸
                 stride=1,
                 padding=None,
                 reduction=16):
        super().__init__()
        assert in_channels % 4 == 0, "in_channels must be divisible by 4"
        self.in_channels = in_channels
        self.branch_channels = in_channels // 4

        # 分支1: 7x7 深度可分离卷积
        self.conv_square = nn.Conv2d(self.branch_channels, self.branch_channels,
                                     kernel_size=kernel_size_small, stride=stride,
                                     padding=kernel_size_small//2 if padding is None else padding,
                                     groups=self.branch_channels, bias=False)
        self.bn_square = nn.BatchNorm2d(self.branch_channels)

        # 分支2: 1x7 + 7x1 串联 (非对称)
        self.conv_asym1_h = nn.Conv2d(self.branch_channels, self.branch_channels,
                                      kernel_size=(1, kernel_size_asym1), stride=stride,
                                      padding=(0, kernel_size_asym1//2), groups=self.branch_channels, bias=False)
        self.conv_asym1_v = nn.Conv2d(self.branch_channels, self.branch_channels,
                                      kernel_size=(kernel_size_asym1, 1), stride=1,
                                      padding=(kernel_size_asym1//2, 0), groups=self.branch_channels, bias=False)
        self.bn_asym1 = nn.BatchNorm2d(self.branch_channels)

        # 分支3: 1x13 + 13x1 串联 (更大感受野)
        self.conv_asym2_h = nn.Conv2d(self.branch_channels, self.branch_channels,
                                      kernel_size=(1, kernel_size_asym2), stride=stride,
                                      padding=(0, kernel_size_asym2//2), groups=self.branch_channels, bias=False)
        self.conv_asym2_v = nn.Conv2d(self.branch_channels, self.branch_channels,
                                      kernel_size=(kernel_size_asym2, 1), stride=1,
                                      padding=(kernel_size_asym2//2, 0), groups=self.branch_channels, bias=False)
        self.bn_asym2 = nn.BatchNorm2d(self.branch_channels)

        # 分支4: 旋转自适应卷积
        self.rot_conv = AngleAdaptiveConv(self.branch_channels, kernel_size=kernel_size_small, stride=stride)

        # 激活函数
        self.act = nn.SiLU()

        # 融合部分：通道注意力 + 1x1卷积
        self.se = ChannelAttention(in_channels, reduction=reduction)
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        ch = self.branch_channels

        # 按通道分割
        x1 = x[:, :ch, :, :]          # 分支1输入
        x2 = x[:, ch:2*ch, :, :]      # 分支2输入
        x3 = x[:, 2*ch:3*ch, :, :]    # 分支3输入
        x4 = x[:, 3*ch:4*ch, :, :]    # 分支4输入

        # 分支1: 7x7 depthwise conv
        out1 = self.conv_square(x1)
        out1 = self.bn_square(out1)
        out1 = self.act(out1)

        # 分支2: 1x7 -> 7x1 (串联)
        out2 = self.conv_asym1_h(x2)
        out2 = self.conv_asym1_v(out2)
        out2 = self.bn_asym1(out2)
        out2 = self.act(out2)

        # 分支3: 1x13 -> 13x1 (串联)
        out3 = self.conv_asym2_h(x3)
        out3 = self.conv_asym2_v(out3)
        out3 = self.bn_asym2(out3)
        out3 = self.act(out3)

        # 分支4: 旋转自适应
        out4 = self.rot_conv(x4)   # 内部已包含激活？原始没有，我们加一个激活
        out4 = self.act(out4)

        # 拼接
        out = torch.cat([out1, out2, out3, out4], dim=1)

        # 融合 (通道注意力 + 1x1)
        out = self.se(out)
        out = self.fusion_conv(out)
        out = self.bn_fusion(out)
        out = self.act(out)

        # 残差连接
        return out + x


# ========== FPN 包装 ==========
@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """
    增强版 FPN，当 fusion_mode='afe' 时使用 MultiBranchAFE。
    支持配置各分支的卷积核尺寸等参数。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes: List[str],
                 afe_kernel_small=7,
                 afe_kernel_asym1=7,
                 afe_kernel_asym2=13,
                 afe_reduction=16,
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
                assert out_channels % 4 == 0, "out_channels must be divisible by 4 for AFE"
                self.dynamic_convs.append(
                    MultiBranchAFE(
                        in_channels=out_channels,
                        kernel_size_small=afe_kernel_small,
                        kernel_size_asym1=afe_kernel_asym1,
                        kernel_size_asym2=afe_kernel_asym2,
                        reduction=afe_reduction
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

        # 输出层（与原FPN相同）
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # 生成额外层
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