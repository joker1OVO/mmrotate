# afe_angle_minimal.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List, Optional


class AngleFreqEnhance(nn.Module):
    """
    基于原版 afe7595 的修改：将半径分组改为角度分组。
    - 权重形状 (n_angles,)
    - 高频掩码保留
    - 其他逻辑（投影、残差等）与原版一致
    """

    def __init__(self,
                 in_channels: int = 256,
                 n_angles: int = 8,
                 high_freq_ratio: float = 0.3,
                 learnable_weights: bool = True,
                 enhance_init: float = 1.0,
                 residual: bool = True,
                 c_mid: int = 16,
                 eps: float = 1e-8):
        super().__init__()
        self.n_angles = n_angles
        self.high_freq_ratio = high_freq_ratio
        self.residual = residual
        self.c_mid = c_mid
        self.eps = eps

        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)

        if learnable_weights:
            self.angle_weights = nn.Parameter(torch.full((n_angles,), enhance_init))
        else:
            self.register_buffer('angle_weights', torch.full((n_angles,), enhance_init))

        self.register_buffer('angle_idx', None)
        self.register_buffer('high_freq_mask', None)

    def _set_masks(self, H: int, W: int, device: torch.device):
        if self.angle_idx is None or self.angle_idx.shape[-2:] != (H, W):
            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(torch.arange(H, device=device),
                                  torch.arange(W, device=device), indexing='ij')
            r = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            theta = torch.atan2(y - cy, x - cx) + math.pi

            # 角度分组索引（均匀划分）
            angle_step = 2 * math.pi / self.n_angles
            angle_idx = (theta / angle_step).floor().long() % self.n_angles

            # 高频掩码（与原版相同）
            r_max = max(cy, cx) if max(cy, cx) > 0 else 1
            high_freq_mask = (r > self.high_freq_ratio * r_max)

            self.angle_idx = angle_idx.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
            self.high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_proj = self.proj_in(x)

        self._set_masks(H, W, x.device)

        # 扩展掩码到 batch 和 c_mid 通道
        a_idx = self.angle_idx.expand(B, self.c_mid, -1, -1)   # [B, c_mid, H, W]
        hf_mask = self.high_freq_mask.expand(B, self.c_mid, -1, -1)

        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # 应用角度权重（与原版相同的 torch.where 逻辑）
        weights = self.angle_weights[a_idx]   # [B, c_mid, H, W]
        gain = torch.where(hf_mask, weights, torch.ones_like(weights))

        x_fft_shift = x_fft_shift * gain

        x_fft_ishift = torch.fft.ifftshift(x_fft_shift, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real
        x_enh = self.proj_out(x_enh)

        return x + x_enh if self.residual else x_enh


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 enhance_levels: List[int] = [0, 1, 2, 3],
                 afe_cfg: dict = dict(n_angles=8, c_mid=16, high_freq_ratio=0.3),
                 **kwargs):
        super(AngleFreqEnhanceFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        self.enhance_levels = enhance_levels
        self.afe_modules = nn.ModuleList()
        for i in range(len(in_channels)):
            if i in enhance_levels:
                self.afe_modules.append(AngleFreqEnhance(in_channels=out_channels, **afe_cfg))
            else:
                self.afe_modules.append(nn.Identity())

    @auto_fp16()
    def forward(self, inputs):
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            feat = lateral_conv(inputs[i + self.start_level])
            feat = self.afe_modules[i](feat)
            laterals.append(feat)

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i - 1] = laterals[i - 1] + upsampled

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