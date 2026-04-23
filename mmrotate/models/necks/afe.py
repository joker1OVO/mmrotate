import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List

# ========== 批量角度估计（与 FAA 一致） ==========
def estimate_main_direction_batch(patch_tensor, eps=1e-8):
    Bn, _, m, _ = patch_tensor.shape
    device = patch_tensor.device
    x_fft = torch.fft.fft2(patch_tensor.squeeze(1), norm='ortho')
    x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
    mag = x_fft_shifted.abs() + eps

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
    B, C, H, W = x.shape
    x_mean = x.mean(dim=1, keepdim=True)
    pad = window_size // 2
    x_pad = F.pad(x_mean, (pad, pad, pad, pad), mode='reflect')
    patches = F.unfold(x_pad, kernel_size=window_size, stride=1)
    N = patches.shape[2]
    patches = patches.transpose(1, 2).reshape(B * N, 1, window_size, window_size)
    angles = estimate_main_direction_batch(patches)  # [B*N]
    angle_map = angles.view(B, H, W)
    return angle_map


class DynamicDirectionalConv(nn.Module):
    def __init__(self, in_channels, mid_channels=16, kernel_size=7, stride=1, padding=3):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 通道降维与升维
        self.reduce = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.expand = nn.Conv2d(mid_channels, in_channels, 1, bias=False)

        # 预定义4个基础核（0°,45°,90°,135°）
        base_kernels = self._create_base_kernels(kernel_size)  # [4, 1, K, K]
        self.register_buffer('base_kernels', base_kernels)

        # 权重预测网络：输入角度特征（sin2θ, cos2θ）
        self.weight_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Softmax(dim=-1)
        )

    def _create_base_kernels(self, k):
        sigma1, sigma2 = 2.5, 1.0
        angles = [0, math.pi/4, math.pi/2, 3*math.pi/4]
        kernels = []
        for theta in angles:
            ax = torch.linspace(-(k//2), k//2, k)
            xx, yy = torch.meshgrid(ax, ax, indexing='ij')
            x_rot = xx * math.cos(theta) + yy * math.sin(theta)
            y_rot = -xx * math.sin(theta) + yy * math.cos(theta)
            kernel = torch.exp(-(x_rot**2 / (2*sigma1**2) + y_rot**2 / (2*sigma2**2)))
            kernel = kernel / kernel.sum()
            kernels.append(kernel)
        return torch.stack(kernels, dim=0).unsqueeze(1)  # [4,1,K,K]

    def forward(self, x, angle_map):
        B, C, H, W = x.shape
        device = x.device

        # 降维
        x_low = self.reduce(x)               # [B, mid, H, W]
        mid = self.mid_channels

        # 生成角度特征
        sin2 = torch.sin(2 * angle_map)      # [B, H, W]
        cos2 = torch.cos(2 * angle_map)
        angle_feat = torch.stack([sin2, cos2], dim=-1)  # [B, H, W, 2]

        # 预测组合权重
        weights = self.weight_net(angle_feat)  # [B, H, W, 4]

        # 组合基础核 -> [B, H, W, K, K]
        base = self.base_kernels.squeeze(1)    # [4, K, K]
        combined = torch.einsum('bhwk,kij->bhwij', weights, base)  # [B, H, W, K, K]

        # 准备滑动窗口和核展平
        x_pad = F.pad(x_low, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        patches = F.unfold(x_pad, kernel_size=self.kernel_size, stride=self.stride)  # [B, mid*K*K, N]
        N = patches.shape[-1]  # H*W
        patches = patches.view(B, mid, self.kernel_size*self.kernel_size, N)  # [B, mid, 49, N]
        patches = patches.permute(0, 1, 3, 2)  # [B, mid, N, 49]

        kernels_flat = combined.view(B, H*W, -1)  # [B, N, 49]

        # 动态深度卷积
        out_low = torch.einsum('bcnk,bnk->bcn', patches, kernels_flat)  # [B, mid, N]
        out_low = out_low.view(B, mid, H, W)

        # 升维输出
        out = self.expand(out_low)  # [B, C, H, W]
        return out


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes: List[str],
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
                self.dynamic_convs.append(DynamicDirectionalConv(out_channels, mid_channels=16))
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

            if mode == 'afe':
                angle_map = compute_angle_map(laterals[i-1])
                enhanced_low = self.dynamic_convs[fusion_idx](laterals[i-1], angle_map)
                up_high = F.interpolate(laterals[i], size=enhanced_low.shape[-2:], **self.upsample_cfg)
                laterals[i-1] = enhanced_low + up_high
            else:  # 'add'
                up_high = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], **self.upsample_cfg)
                laterals[i-1] = laterals[i-1] + up_high

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # 额外层生成
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