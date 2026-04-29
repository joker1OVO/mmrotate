# afe_fixed_gain.py
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
    极简固定增益模块：仅对高频区域幅度谱乘以固定系数（如1.2），无角度/半径分组，无可学习参数。
    """

    def __init__(self,
                 in_channels: int = 256,
                 c_mid: int = 16,
                 high_freq_ratio: float = 0.3,
                 fixed_gain: float = 1.2,
                 residual: bool = True,
                 eps: float = 1e-8):
        super().__init__()
        self.c_mid = c_mid
        self.high_freq_ratio = high_freq_ratio
        self.fixed_gain = fixed_gain
        self.residual = residual
        self.eps = eps

        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)
        # 初始化投影层（可选，保持与 baseline 一致）
        nn.init.kaiming_normal_(self.proj_in.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj_out.weight, mode='fan_out', nonlinearity='relu')

        self.register_buffer('high_freq_mask', None)
        self._cached_HW = None

    def _build_mask(self, H: int, W: int, device: torch.device):
        """生成高频掩码（仅半径大于 high_freq_ratio * max_r 的区域）"""
        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing='ij')
        r = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_r = max(cy, cx)
        high_freq_mask = (r > self.high_freq_ratio * max_r)
        self.high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        self._cached_HW = (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # 投影降维
        x_proj = self.proj_in(x)   # [B, c_mid, H, W]

        # 构建高频掩码
        if self._cached_HW != (H, W):
            self._build_mask(H, W, device)

        # FFT
        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs() + self.eps

        # 应用固定增益到高频区域，低频区域增益=1
        hf_mask = self.high_freq_mask.expand(B, self.c_mid, -1, -1)  # [B,c_mid,H,W]
        gain = torch.where(hf_mask, torch.tensor(self.fixed_gain, device=device), torch.ones_like(mag))
        mag_enhanced = mag * gain

        # 保持相位，重构复数谱
        x_fft_shift_enhanced = mag_enhanced * torch.exp(1j * torch.angle(x_fft_shift))
        x_fft_ishift = torch.fft.ifftshift(x_fft_shift_enhanced, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real

        # 投影回原通道
        x_enh = self.proj_out(x_enh)

        if self.residual:
            return x + x_enh
        else:
            return x_enh


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 enhance_levels: List[int] = [0, 1, 2, 3],
                 afe_cfg: dict = dict(c_mid=16, high_freq_ratio=0.3, fixed_gain=1.2, residual=True),
                 **kwargs):
        super().__init__(
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