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
    频域角度增强模块（全频段学习，带数值稳定化措施）
    主要改进：
    1. 排除直流分量 (rho < 0.5) 不参与加权
    2. 增益采用 1 + tanh(gain_raw) 限制范围 (0,2)
    3. 可选输出 clamp 防止数值溢出
    4. 支持半径-角度极坐标分组，可学习权重
    5. 可选的 high_freq_ratio 掩码（若设为0则全频段学习）
    """

    def __init__(self,
                 in_channels: int = 256,
                 c_mid: int = 16,
                 n_angles: int = 8,
                 radius_width: int = 8,
                 overlap_ratio: float = 1.5,
                 high_freq_ratio: float = 0.3,      # 0 表示全频段；>0 表示只增强高频部分
                 learnable_weights: bool = True,
                 residual: bool = True,
                 use_hann_window: bool = False,
                 out_clip: float = 10.0,            # None 表示不裁剪
                 eps: float = 1e-8):
        super().__init__()
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

        # 可学习权重：形状 (c_mid, n_angles, n_radii)，但 n_radii 动态确定
        if learnable_weights:
            # 注意：这里暂时用 placeholder，在 _build_masks 中会根据实际 n_radii 调整
            self.weights_raw = nn.Parameter(torch.ones(c_mid, n_angles, 1))
        else:
            self.register_buffer('weights_raw', torch.ones(c_mid, n_angles, 1))

        # 缓存
        self.register_buffer('radius_idx', None)
        self.register_buffer('angle_weights', None)
        self.register_buffer('high_freq_mask', None)   # 可选高频掩码
        self.register_buffer('_hann_window', None)
        self._cached_HW = None
        self._cached_n_radii = None

    def _build_masks(self, H: int, W: int, device: torch.device):
        """生成半径索引、角度软分配权重、高频掩码（可选）"""
        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing='ij')
        r = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2).float()
        theta = torch.atan2(y - cy, x - cx) + math.pi   # [0, 2π)

        # 半径均匀分区
        max_r = max(cy, cx)
        n_radii = int(max_r // self.radius_width) + 1
        radius_idx = torch.floor(r / self.radius_width).long().clamp(0, n_radii - 1)

        # 角度软分配（三角加权，覆盖 [0, π) 即可）
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

        # 可选高频掩码: 半径 > high_freq_ratio * max_r 的区域视为高频
        if self.high_freq_ratio > 0:
            high_freq_mask = (r > self.high_freq_ratio * max_r)
        else:
            # 若 high_freq_ratio <= 0，不限制（全频段）
            high_freq_mask = torch.ones_like(r, dtype=torch.bool)

        # 排除直流分量（半径 < 0.5 的点）不参与加权（增益固定为1）
        dc_mask = (r < 0.5)
        # 最终有效掩码：不是直流点，并且（如果启用高频掩码则还需满足高频条件，否则不过滤）
        if self.high_freq_ratio > 0:
            valid_mask = ~dc_mask & high_freq_mask
        else:
            valid_mask = ~dc_mask

        # 存储
        self.radius_idx = radius_idx.unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
        self.angle_weights = angle_weights.unsqueeze(0)                # [1, n_angles, H, W]
        self.valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)         # [1,1,H,W]
        self._cached_HW = (H, W)
        self._cached_n_radii = n_radii

        # 调整可学习权重的维度
        if hasattr(self, 'weights_raw') and self.weights_raw is not None:
            if self.weights_raw.size(-1) != n_radii:
                new_weights = torch.ones(self.c_mid, self.n_angles, n_radii, device=device)
                if isinstance(self.weights_raw, nn.Parameter):
                    self.weights_raw = nn.Parameter(new_weights)
                else:
                    self.weights_raw = new_weights

        # 可选加窗
        if self.use_hann_window and self._hann_window is None:
            hann_h = torch.hann_window(H, device=device)
            hann_w = torch.hann_window(W, device=device)
            hann_2d = hann_h.unsqueeze(1) * hann_w.unsqueeze(0)
            self._hann_window = hann_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        x_proj = self.proj_in(x)   # [B, c_mid, H, W]

        # 可选加窗
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
        valid_mask = self.valid_mask.expand(B, -1, H, W)     # [B,1,H,W]
        n_radii = self._cached_n_radii

        # 增益计算: 增益 = 1 + tanh(gain_raw) ，范围 (0,2)
        # 先初始化增益为全1
        gain = torch.ones(B, self.c_mid, H, W, device=device)

        # 只在有效掩码区域计算加权增益
        # 对每个通道独立计算
        for c in range(self.c_mid):
            # 提取该通道的原始权重 (n_angles, n_radii)
            w_c_raw = self.weights_raw[c]  # [n_angles, n_radii]
            # 应用 tanh 约束，得到限制在 (0,2) 的增益基底
            w_c = 1 + torch.tanh(w_c_raw)  # 范围 (0,2)

            # 初始化该通道的增益图
            gain_c = torch.zeros(B, H, W, device=device)

            # 对每个半径环累加
            for r in range(n_radii):
                mask_r = (r_idx == r).float().squeeze(1)            # [B, H, W]
                w_r = w_c[:, r]                                    # [n_angles]
                # 角度权重加权求和
                weighted_angles = (aw * w_r.view(1, -1, 1, 1)).sum(dim=1)   # [B, H, W]
                gain_c += weighted_angles * mask_r

            # 有效区域替换增益，无效区域保持1
            gain[:, c, :, :] = torch.where(valid_mask.squeeze(1), gain_c, torch.ones_like(gain_c))

        # 应用增益到幅度谱
        mag_enhanced = mag * gain
        x_fft_shift_enhanced = mag_enhanced * torch.exp(1j * torch.angle(x_fft_shift))

        # 逆变换
        x_fft_ishift = torch.fft.ifftshift(x_fft_shift_enhanced, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real
        x_enh = self.proj_out(x_enh)

        # 可选输出裁剪
        if self.out_clip is not None:
            x_enh = torch.clamp(x_enh, -self.out_clip, self.out_clip)

        if self.residual:
            # 注意：x_enh 可能数值范围较大，但残差连接和之前的 clamp 已做保护
            return x + x_enh
        else:
            return x_enh


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """在 FPN 的侧向连接处添加频域角度增强模块"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 enhance_levels: Optional[List[int]] = None,
                 afe_cfg: dict = dict(
                     c_mid=16, n_angles=8, radius_width=8, overlap_ratio=1.5,
                     high_freq_ratio=0.3, learnable_weights=True, residual=True,
                     use_hann_window=False, out_clip=10.0),
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
        # 侧向连接 + AFE 增强
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            feat = lateral_conv(inputs[i + self.start_level])
            feat = self.afe_modules[i](feat)
            laterals.append(feat)

        # 自顶向下融合（原始 FPN 逻辑）
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 构建输出层
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # 额外层（P6, P7 ...）
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