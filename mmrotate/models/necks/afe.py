# afe_fixed.py
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
    频域极坐标增强模块（修复版）：
    - 均匀半径分区 + 重叠角度扇区
    - 通道独立可学习权重
    - 高频掩码：只调制半径 > high_freq_ratio * max_r 的区域，低频和直流保持 1
    - 可选权重范围约束（tanh 映射）
    """

    def __init__(self,
                 in_channels: int = 256,
                 c_mid: int = 16,
                 n_angles: int = 8,
                 radius_width: int = 8,
                 high_freq_ratio: float = 0.8,      # 新增：高频比例，默认 0.8 只增强最外圈
                 overlap_ratio: float = 1.5,
                 learnable_weights: bool = True,
                 weight_range: float = 0.5,         # 新增：权重范围 [1-weight_range, 1+weight_range]
                 residual: bool = True,
                 use_hann_window: bool = False,
                 eps: float = 1e-8):
        super().__init__()
        self.c_mid = c_mid
        self.n_angles = n_angles
        self.radius_width = radius_width
        self.high_freq_ratio = high_freq_ratio
        self.overlap_ratio = overlap_ratio
        self.residual = residual
        self.use_hann_window = use_hann_window
        self.eps = eps
        self.weight_range = weight_range

        # 投影层
        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)

        if learnable_weights:
            # 权重形状: (c_mid, n_angles, n_radii)
            # 初始化为0（映射后为1）或1？为了配合 tanh 限制，初始为0
            self.weights = nn.Parameter(torch.zeros(c_mid, n_angles, 1))
        else:
            self.register_buffer('weights', torch.zeros(c_mid, n_angles, 1))

        # 缓存掩码
        self.register_buffer('radius_idx', None)
        self.register_buffer('angle_weights', None)
        self.register_buffer('high_freq_mask', None)   # 新增
        self.register_buffer('_hann_window', None)
        self._cached_HW = None
        self._cached_n_radii = None

    def _build_masks(self, H: int, W: int, device: torch.device):
        """生成半径索引、角度软分配权重、高频掩码"""
        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing='ij')
        r = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        theta = torch.atan2(y - cy, x - cx) + math.pi   # [0, 2π)

        # 半径索引（均匀分区）
        max_r = max(cy, cx)
        n_radii = int(max_r // self.radius_width) + 1
        radius_idx = torch.floor(r / self.radius_width).long().clamp(0, n_radii - 1)

        # 角度软分配（三角加权，覆盖 [0, π) 以保持对称性）
        theta = theta % math.pi   # 映射到 [0, π)
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

        # 高频掩码（半径大于 high_freq_ratio * max_r 的点）
        high_freq_mask = (r > self.high_freq_ratio * max_r)

        # 缓存
        self.radius_idx = radius_idx.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        self.angle_weights = angle_weights.unsqueeze(0)          # [1, n_angles, H, W]
        self.high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        self._cached_HW = (H, W)
        self._cached_n_radii = n_radii

        # 调整权重的最后一维（半径数）
        if self.weights.size(-1) != n_radii:
            new_weights = torch.zeros(self.c_mid, self.n_angles, n_radii, device=device)
            if isinstance(self.weights, nn.Parameter):
                self.weights = nn.Parameter(new_weights)
            else:
                self.weights = new_weights

        if self.use_hann_window and self._hann_window is None:
            hann_1d = torch.hann_window(H, device=device)
            hann_2d = hann_1d.unsqueeze(1) * torch.hann_window(W, device=device).unsqueeze(0)
            self._hann_window = hann_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        x_proj = self.proj_in(x)   # [B, c_mid, H, W]

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

        r_idx = self.radius_idx.expand(B, -1, H, W)          # [B,1,H,W]
        aw = self.angle_weights.expand(B, -1, H, W)          # [B, n_angles, H, W]
        hf_mask = self.high_freq_mask.expand(B, -1, H, W)    # [B,1,H,W] 高频掩码

        n_radii = self._cached_n_radii

        # 计算增益（先初始化为1，然后高频区域替换为学习到的增益）
        gain = torch.ones(B, self.c_mid, H, W, device=device)

        for c in range(self.c_mid):
            w_c = self.weights[c]                     # [n_angles, n_radii]
            # 对每个半径环独立处理
            gain_c = torch.zeros(B, H, W, device=device)
            for r in range(n_radii):
                mask_r = (r_idx == r).float().squeeze(1)   # [B, H, W]
                # 获取该半径环对应的角度权重（通过 soft 分配）
                # 取 w_c 的第 r 列，形状 [n_angles]
                w_r = w_c[:, r]                            # [n_angles]
                # 计算每个空间位置的角度加权和
                # aw: [B, n_angles, H, W], w_r: [n_angles] -> broadcast multiply
                weighted_angle = (aw * w_r.view(1, -1, 1, 1)).sum(dim=1)  # [B, H, W]
                gain_c = gain_c + weighted_angle * mask_r
            # 应用高频掩码：高频区域用 gain_c，低频区域保持 1
            # 注意：gain_c 初始值来自可学习权重，可能不是 1，需要先限制范围（可选）
            if self.weight_range > 0:
                # 通过 tanh 将权重的有效值限制在 [1-weight_range, 1+weight_range]
                # 注意：gain_c 中的值来自 weights 的直接输出，weights 初始为0，所以初始 gain_c=0
                # 但我们希望初始增益为1，所以需要加偏移和缩放
                # 更简单：将 weights 作为缩放增量，gain_c = 1 + weight_range * tanh(original_gain)
                # 但这里的 gain_c 是经过半径和角度聚合后的值，直接映射会破坏结构。
                # 改为在权重加载后映射：先计算 raw_gain = gain_c，然后应用映射。
                raw_gain = gain_c
                gain_c = 1.0 + self.weight_range * torch.tanh(raw_gain)
            # 高频区域使用 gain_c，低频区域保持1
            gain_c_masked = torch.where(hf_mask.squeeze(1), gain_c, torch.ones_like(gain_c))
            gain[:, c, :, :] = gain_c_masked

        # 应用增益到幅度谱
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
                 enhance_levels: List[int] = [0, 1, 2, 3],
                 afe_cfg: dict = dict(
                     c_mid=16,
                     n_angles=8,
                     radius_width=8,
                     high_freq_ratio=0.8,
                     overlap_ratio=1.5,
                     learnable_weights=True,
                     weight_range=0.5,
                     residual=True,
                     use_hann_window=False,
                 ),
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