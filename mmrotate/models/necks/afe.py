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
    修复版：仅在高频区域按角度分组加权，低频和直流保持不变。
    """

    def __init__(self,
                 in_channels: int = 256,
                 c_mid: int = 16,
                 n_angles: int = 8,
                 high_freq_ratio: float = 0.3,
                 learnable_weights: bool = True,
                 residual: bool = True,
                 eps: float = 1e-8):
        super().__init__()
        self.c_mid = c_mid
        self.n_angles = n_angles
        self.high_freq_ratio = high_freq_ratio
        self.residual = residual
        self.eps = eps

        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj_in.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj_out.weight, mode='fan_out', nonlinearity='relu')

        if learnable_weights:
            self.angle_weights = nn.Parameter(torch.ones(c_mid, n_angles))
        else:
            self.register_buffer('angle_weights', torch.ones(c_mid, n_angles))

        self.register_buffer('angle_idx', None)
        self.register_buffer('high_freq_mask', None)
        self._cached_HW = None

    def _build_masks(self, H: int, W: int, device: torch.device):
        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing='ij')
        r = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        theta = torch.atan2(y - cy, x - cx) + math.pi

        max_r = max(cy, cx)
        high_freq_mask = (r > self.high_freq_ratio * max_r)

        angle_step = 2 * math.pi / self.n_angles
        angle_idx = (theta / angle_step).floor().long() % self.n_angles

        self.angle_idx = angle_idx.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        self.high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        self._cached_HW = (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        x_proj = self.proj_in(x)   # [B, c_mid, H, W]

        if self._cached_HW != (H, W):
            self._build_masks(H, W, device)

        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs() + self.eps

        # 角度索引和高频掩码
        angle_idx = self.angle_idx.expand(B, self.c_mid, H, W)   # [B, c_mid, H, W]
        hf_mask = self.high_freq_mask.expand(B, 1, H, W)         # [B,1,H,W]

        # 获取增益：只对高频区域应用角度权重，低频区域增益=1
        # 方法：先初始化为全1，然后在高频区域用角度权重替换
        gain = torch.ones(B, self.c_mid, H, W, device=device)
        # 对于每个通道，从 angle_weights 中取对应的权重
        for c in range(self.c_mid):
            # 取出该通道的角度权重向量 [n_angles]
            w = self.angle_weights[c]  # [n_angles]
            # 根据 angle_idx 取值，得到 [B, H, W]
            gain_c = w[angle_idx[:, c, :, :]]  # 注意 angle_idx 是 [B, c_mid, H, W]
            # 应用高频掩码：高频区域使用 gain_c，低频区域保持1
            gain_c = torch.where(hf_mask.squeeze(1), gain_c, torch.ones_like(gain_c))
            gain[:, c, :, :] = gain_c

        mag_enhanced = mag * gain
        x_fft_shift_enhanced = mag_enhanced * torch.exp(1j * torch.angle(x_fft_shift))
        x_fft_ishift = torch.fft.ifftshift(x_fft_shift_enhanced, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real
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
                 enhance_levels: Optional[List[int]] = None,
                 afe_cfg: dict = dict(c_mid=16, n_angles=8, high_freq_ratio=0.3,
                                      learnable_weights=True, residual=True),
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        num_lateral = len(self.lateral_convs)
        self.afe_modules = nn.ModuleList()
        for i in range(num_lateral):
            if enhance_levels is None or i in enhance_levels:
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