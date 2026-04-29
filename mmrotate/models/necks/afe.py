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
    频域角度增强模块（支持恒等模式调试）
    若 identity=True，模块不修改特征，直接返回输入。
    """

    def __init__(self,
                 in_channels: int = 256,
                 c_mid: int = 16,
                 n_angles: int = 8,
                 radius_width: int = 8,
                 overlap_ratio: float = 1.5,
                 high_freq_ratio: float = 0.3,
                 learnable_weights: bool = True,
                 residual: bool = True,
                 use_hann_window: bool = False,
                 out_clip: float = 10.0,
                 identity: bool = False,          # 新增恒等模式开关
                 eps: float = 1e-8):
        super().__init__()
        if identity:
            # 恒等模式：不创建任何参数，forward 直接返回 x
            self.identity = True
            return

        self.identity = False
        self.in_channels = in_channels
        self.c_mid = c_mid
        self.n_angles = n_angles
        self.radius_width = radius_width
        self.overlap_ratio = overlap_ratio
        self.high_freq_ratio = high_freq_ratio
        self.residual = residual
        self.use_hann_window = use_hann_window
        self.out_clip = out_clip
        self.eps = eps

        # 投影层
        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj_in.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj_out.weight, mode='fan_out', nonlinearity='relu')

        # 可学习权重：形状 (c_mid, n_angles, n_radii)，n_radii 动态调整
        if learnable_weights:
            self.weights_raw = nn.Parameter(torch.ones(c_mid, n_angles, 1))
        else:
            self.register_buffer('weights_raw', torch.ones(c_mid, n_angles, 1))

        # 缓存
        self.register_buffer('radius_idx', None)
        self.register_buffer('angle_weights', None)
        self.register_buffer('high_freq_mask', None)
        self.register_buffer('_hann_window', None)
        self._cached_HW = None
        self._cached_n_radii = None

    def _build_masks(self, H: int, W: int, device: torch.device):
        """生成半径索引、角度软分配权重、高频掩码（可选）"""
        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing='ij')
        r = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2).float()
        theta = torch.atan2(y - cy, x - cx) + math.pi

        max_r = max(cy, cx)
        n_radii = int(max_r // self.radius_width) + 1
        radius_idx = torch.floor(r / self.radius_width).long().clamp(0, n_radii - 1)

        theta = theta % math.pi
        delta = math.pi / self.n_angles
        half_width = self.overlap_ratio * delta / 2.0
        angle_weights = torch.zeros(self.n_angles, H, W, device=device)
        for a in range(self.n_angles):
            center = a * delta + delta / 2.0
            dist = (theta - center).abs()
            mask = dist < half_width
            w = (1 - dist / half_width).clamp(min=0)
            angle_weights[a] = w * mask.float()
        sum_w = angle_weights.sum(dim=0, keepdim=True) + self.eps
        angle_weights = angle_weights / sum_w

        # 高频掩码（可选）
        if self.high_freq_ratio > 0:
            high_freq_mask = (r > self.high_freq_ratio * max_r)
        else:
            high_freq_mask = torch.ones_like(r, dtype=torch.bool)

        # 排除直流分量（半径 < 0.5）
        dc_mask = (r < 0.5)
        valid_mask = ~dc_mask & high_freq_mask

        self.radius_idx = radius_idx.unsqueeze(0).unsqueeze(0)
        self.angle_weights = angle_weights.unsqueeze(0)
        self.valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
        self._cached_HW = (H, W)
        self._cached_n_radii = n_radii

        # 调整权重尺寸
        if hasattr(self, 'weights_raw') and self.weights_raw is not None:
            if self.weights_raw.size(-1) != n_radii:
                new_weights = torch.ones(self.c_mid, self.n_angles, n_radii, device=device)
                if isinstance(self.weights_raw, nn.Parameter):
                    self.weights_raw = nn.Parameter(new_weights)
                else:
                    self.weights_raw = new_weights

        if self.use_hann_window and self._hann_window is None:
            hann_h = torch.hann_window(H, device=device)
            hann_w = torch.hann_window(W, device=device)
            self._hann_window = (hann_h.unsqueeze(1) * hann_w.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return x

        B, C, H, W = x.shape
        device = x.device

        x_proj = self.proj_in(x)

        if self.use_hann_window:
            if self._hann_window is None or self._hann_window.shape[-2:] != (H, W):
                self._build_masks(H, W, device)
            x_proj = x_proj * self._hann_window

        # FFT
        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs() + self.eps

        if self._cached_HW != (H, W):
            self._build_masks(H, W, device)

        r_idx = self.radius_idx.expand(B, -1, H, W)
        aw = self.angle_weights.expand(B, -1, H, W)
        valid_mask = self.valid_mask.expand(B, -1, H, W)
        n_radii = self._cached_n_radii

        # 增益计算，范围限制 (0,2)
        gain = torch.ones(B, self.c_mid, H, W, device=device)
        for c in range(self.c_mid):
            w_c_raw = self.weights_raw[c]                     # [n_angles, n_radii]
            w_c = 1 + torch.tanh(w_c_raw)                     # (0,2)
            gain_c = torch.zeros(B, H, W, device=device)
            for r in range(n_radii):
                mask_r = (r_idx == r).float().squeeze(1)
                w_r = w_c[:, r]                               # [n_angles]
                weighted_angles = (aw * w_r.view(1, -1, 1, 1)).sum(dim=1)
                gain_c += weighted_angles * mask_r
            gain[:, c, :, :] = torch.where(valid_mask.squeeze(1), gain_c, torch.ones_like(gain_c))

        mag_enhanced = mag * gain
        x_fft_shift_enhanced = mag_enhanced * torch.exp(1j * torch.angle(x_fft_shift))

        x_fft_ishift = torch.fft.ifftshift(x_fft_shift_enhanced, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real
        x_enh = self.proj_out(x_enh)

        if self.out_clip is not None:
            x_enh = torch.clamp(x_enh, -self.out_clip, self.out_clip)

        if self.residual:
            return x + x_enh
        else:
            return x_enh


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """在 FPN 的侧向连接处添加频域角度增强模块（支持恒等模式）"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 enhance_levels: Optional[List[int]] = None,
                 afe_cfg: dict = dict(
                     c_mid=16, n_angles=8, radius_width=8, overlap_ratio=1.5,
                     high_freq_ratio=0.3, learnable_weights=True, residual=True,
                     use_hann_window=False, out_clip=10.0, identity=False),
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        num_lateral = len(self.lateral_convs)
        self.afe_modules = nn.ModuleList()
        for i in range(num_lateral):
            if enhance_levels is not None and i not in enhance_levels:
                self.afe_modules.append(nn.Identity())
            else:
                self.afe_modules.append(AngleFreqEnhance(in_channels=out_channels, **afe_cfg))

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