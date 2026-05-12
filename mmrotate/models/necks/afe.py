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


# ========== 旋转自适应卷积模块（预旋转60角度）==========
class AngleAdaptiveConv(nn.Module):
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

        angle_fft = compute_angle_map(x, window_size=k, stride=self.stride)
        angle = angle_fft + self.angle_bias + math.pi / 2
        angle = angle % math.pi
        angle = torch.clamp(angle, 0, math.pi - 1e-6)

        pre_rotated = []
        for i in range(self.num_angles):
            theta_i = self.angle_grid[i]
            rot_k = self._rotate_kernel(self.base_kernel, theta_i)
            pre_rotated.append(rot_k.view(C, k*k))
        pre_rotated = torch.stack(pre_rotated, dim=0)

        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        patches = F.unfold(x_pad, kernel_size=k, stride=self.stride)
        N = patches.shape[-1]
        patches = patches.view(B, C, k*k, N).permute(0, 1, 3, 2)
        patches_flat = patches.permute(0, 2, 1, 3).reshape(B*N, C, k*k)

        angle_flat = angle.reshape(-1)
        indices = torch.argmin(torch.abs(angle_flat.unsqueeze(1) - self.angle_grid.unsqueeze(0)), dim=1)
        selected_kernels = pre_rotated[indices]
        out_flat = (patches_flat * selected_kernels).sum(dim=2)
        out = out_flat.view(B, C, H, W)
        return out


# ========== 通道注意力 (SE) ==========
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
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ========== 空间注意力 ==========
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)   # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,H,W]
        concat = torch.cat([avg_out, max_out], dim=1)   # [B,2,H,W]
        attention = self.sigmoid(self.conv(concat))     # [B,1,H,W]
        return x * attention


# ========== 四分支主模块（带 GroupNorm 和 CBAM 风格注意力）==========
class MultiBranchAFE(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size_small=7,
                 kernel_size_asym1=7,
                 kernel_size_asym2=13,
                 stride=1,
                 padding=None,
                 reduction=16,          # SE 压缩比
                 spatial_kernel=7):     # 空间注意力卷积核大小
        super().__init__()
        assert in_channels % 4 == 0, "in_channels must be divisible by 4"
        self.in_channels = in_channels
        self.branch_channels = in_channels // 4

        # 分支1: 7x7 depthwise conv
        self.conv_square = nn.Conv2d(self.branch_channels, self.branch_channels,
                                     kernel_size=kernel_size_small, stride=stride,
                                     padding=kernel_size_small//2 if padding is None else padding,
                                     groups=self.branch_channels, bias=False)
        self.gn_square = nn.GroupNorm(num_groups=16, num_channels=self.branch_channels)

        # 分支2: 1x7 + 7x1 (非对称)
        self.conv_asym1_h = nn.Conv2d(self.branch_channels, self.branch_channels,
                                      kernel_size=(1, kernel_size_asym1), stride=stride,
                                      padding=(0, kernel_size_asym1//2), groups=self.branch_channels, bias=False)
        self.conv_asym1_v = nn.Conv2d(self.branch_channels, self.branch_channels,
                                      kernel_size=(kernel_size_asym1, 1), stride=1,
                                      padding=(kernel_size_asym1//2, 0), groups=self.branch_channels, bias=False)
        self.gn_asym1 = nn.GroupNorm(num_groups=16, num_channels=self.branch_channels)

        # 分支3: 1x13 + 13x1 (更大感受野)
        self.conv_asym2_h = nn.Conv2d(self.branch_channels, self.branch_channels,
                                      kernel_size=(1, kernel_size_asym2), stride=stride,
                                      padding=(0, kernel_size_asym2//2), groups=self.branch_channels, bias=False)
        self.conv_asym2_v = nn.Conv2d(self.branch_channels, self.branch_channels,
                                      kernel_size=(kernel_size_asym2, 1), stride=1,
                                      padding=(kernel_size_asym2//2, 0), groups=self.branch_channels, bias=False)
        self.gn_asym2 = nn.GroupNorm(num_groups=16, num_channels=self.branch_channels)

        # 分支4: 旋转自适应卷积
        self.rot_conv = AngleAdaptiveConv(self.branch_channels, kernel_size=kernel_size_small, stride=stride)

        # 激活函数
        self.act = nn.SiLU()

        # 融合部分: 通道注意力 + 空间注意力 (CBAM风格) → 1x1卷积
        self.channel_att = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.gn_fusion = nn.GroupNorm(num_groups=16, num_channels=in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        ch = self.branch_channels

        # 分割
        x1 = x[:, :ch, :, :]
        x2 = x[:, ch:2*ch, :, :]
        x3 = x[:, 2*ch:3*ch, :, :]
        x4 = x[:, 3*ch:4*ch, :, :]

        # 分支1
        out1 = self.conv_square(x1)
        out1 = self.gn_square(out1)
        out1 = self.act(out1)

        # 分支2
        out2 = self.conv_asym1_h(x2)
        out2 = self.conv_asym1_v(out2)
        out2 = self.gn_asym1(out2)
        out2 = self.act(out2)

        # 分支3
        out3 = self.conv_asym2_h(x3)
        out3 = self.conv_asym2_v(out3)
        out3 = self.gn_asym2(out3)
        out3 = self.act(out3)

        # 分支4
        out4 = self.rot_conv(x4)
        out4 = self.act(out4)

        # 拼接
        out = torch.cat([out1, out2, out3, out4], dim=1)

        # 融合: 通道注意力 → 空间注意力 → GroupNorm → 激活 → 1x1卷积
        out = self.channel_att(out)
        out = self.spatial_att(out)
        out = self.gn_fusion(out)
        out = self.act(out)
        out = self.fusion_conv(out)

        # 残差连接
        return out + x


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """
    增强版 FPN，使用 MultiBranchAFE 作为 'afe' 融合模式。
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
                 afe_spatial_kernel=7,
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
                assert out_channels % 4 == 0, "out_channels must be divisible by 4"
                self.dynamic_convs.append(
                    MultiBranchAFE(
                        in_channels=out_channels,
                        kernel_size_small=afe_kernel_small,
                        kernel_size_asym1=afe_kernel_asym1,
                        kernel_size_asym2=afe_kernel_asym2,
                        reduction=afe_reduction,
                        spatial_kernel=afe_spatial_kernel
                    )
                )
            else:
                self.dynamic_convs.append(None)

    @auto_fp16()
    def forward(self, inputs):
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

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

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