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

class DynamicDirectionalConv(nn.Module):
    """
    方向性卷积模块，支持可配置的 mid_channels 和 kernel_size
    """
    def __init__(self, in_channels, mid_channels=16, kernel_size=7, stride=1, padding=None):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

        self.reduce = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.expand = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.angle_bias = nn.Parameter(torch.tensor(0.0))
        self.fixed_kernel = nn.Parameter(
            torch.randn(mid_channels, 1, kernel_size, kernel_size) * 0.01
        )

    def forward(self, x):
        assert not torch.isnan(x).any(), "NaN in input x"
        B, C, H, W = x.shape
        device = x.device
        k = self.kernel_size
        pad = self.padding
        mid = self.mid_channels
        assert not torch.isnan(self.reduce.weight).any(), "NaN in reduce.weight"
        x_low = self.reduce(x)   # [B, mid, H, W]
        assert not torch.isnan(x_low).any(), "NaN in x_low after reduce"
        # 计算角度图（在低维特征上）
        angle_fft = compute_angle_map(x_low, window_size=k, stride=self.stride)
        angle = angle_fft + self.angle_bias
        angle = angle % math.pi
        assert not torch.isnan(angle).any(), "NaN in angle (after bias and mod)"
        # 滑窗获取 patches
        x_pad = F.pad(x_low, (pad, pad, pad, pad), mode='reflect')
        patches = F.unfold(x_pad, kernel_size=k, stride=self.stride)
        N = patches.shape[-1]
        patches = patches.view(B, mid, k * k, N).permute(0, 1, 3, 2)

        # 恢复成图像形式 [B*N, mid, k, k]
        patches_img = patches.permute(0, 2, 1, 3).reshape(B*N, mid, k, k)
        assert not torch.isnan(patches_img).any(), "NaN in patches_img"
        # 构建仿射变换，旋转 patch
        angle_flat = angle.reshape(-1)
        cos_t = torch.cos(angle_flat)
        sin_t = torch.sin(angle_flat)
        center = (k - 1) / 2.0
        theta_affine = torch.zeros(B * N, 2, 3, device=device)
        theta_affine[:, 0, 0] = cos_t
        theta_affine[:, 0, 1] = -sin_t
        theta_affine[:, 0, 2] = center - cos_t * center + sin_t * center
        theta_affine[:, 1, 0] = sin_t
        theta_affine[:, 1, 1] = cos_t
        theta_affine[:, 1, 2] = center - sin_t * center - cos_t * center

        grid = F.affine_grid(theta_affine, patches_img.size(), align_corners=False)
        patches_rot = F.grid_sample(patches_img, grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=False)
        assert not torch.isnan(patches_rot).any(), "NaN in patches_rot after grid_sample"

        # 深度卷积
        out_conv = F.conv2d(patches_rot, self.fixed_kernel, groups=mid)
        assert not torch.isnan(out_conv).any(), "NaN in out_conv after depthwise conv"

        out_conv = out_conv.view(B, N, mid).permute(0, 2, 1)
        out_low = out_conv.view(B, mid, H, W)
        out = self.expand(out_low)
        assert not torch.isnan(out).any(), "NaN in final out after expand"

        return out


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """
    增强版 FPN，支持可配置的 AFE 模块参数
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes: List[str],
                 afe_mid_channels=16,      # 新增：降维通道数
                 afe_kernel_size=7,        # 新增：局部窗口大小 / 卷积核大小
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
                self.dynamic_convs.append(
                    DynamicDirectionalConv(
                        out_channels,
                        mid_channels=afe_mid_channels,
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