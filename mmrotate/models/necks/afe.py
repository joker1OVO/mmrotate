import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List, Optional


class AngleFreqEnhance(nn.Module):
    """频域角度增强模块（静态卷积层版本，兼容 FLOPs 计算）"""
    def __init__(self,
                 in_channels: int = 256,
                 c_mid: int = 16,
                 n_angles: int = 8,
                 radius_width: int = 8,
                 overlap_ratio: float = 1.5,
                 learnable_weights: bool = True,
                 residual: bool = True,
                 use_hann_window: bool = False,
                 eps: float = 1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.c_mid = c_mid
        self.n_angles = n_angles
        self.radius_width = radius_width
        self.overlap_ratio = overlap_ratio
        self.residual = residual
        self.use_hann_window = use_hann_window
        self.eps = eps

        # ============ 静态创建卷积层（关键修改）============
        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj_in.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj_out.weight, mode='fan_out', nonlinearity='relu')

        if learnable_weights:
            self.weights = nn.Parameter(torch.ones(c_mid, n_angles, 1))
        else:
            self.register_buffer('weights', torch.ones(c_mid, n_angles, 1))

        # 缓冲区，用于缓存半径索引和角度权重（与特征图尺寸相关）
        self.register_buffer('radius_idx', None)
        self.register_buffer('angle_weights', None)
        self.register_buffer('_hann_window', None)
        self._cached_HW = None
        self._cached_n_radii = None

    def _build_masks(self, H: int, W: int, device: torch.device):
        """生成半径索引和角度软分配权重（与之前相同）"""
        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing='ij')
        r = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
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

        self.radius_idx = radius_idx.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        self.angle_weights = angle_weights.unsqueeze(0)         # [1, n_angles, H, W]
        self._cached_HW = (H, W)
        self._cached_n_radii = n_radii

        # 调整可学习权重的维度（如果半径数变化）
        if hasattr(self, 'weights') and self.weights is not None:
            if self.weights.size(-1) != n_radii:
                new_weights = torch.ones(self.c_mid, self.n_angles, n_radii, device=device)
                if isinstance(self.weights, nn.Parameter):
                    self.weights = nn.Parameter(new_weights)
                else:
                    self.weights = new_weights

        if self.use_hann_window and self._hann_window is None:
            hann_h = torch.hann_window(H, device=device)
            hann_w = torch.hann_window(W, device=device)
            hann_2d = hann_h.unsqueeze(1) * hann_w.unsqueeze(0)
            self._hann_window = hann_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # 静态卷积层，直接使用
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

        # 构建掩码
        if self._cached_HW != (H, W):
            self._build_masks(H, W, device)

        r_idx = self.radius_idx.expand(B, -1, H, W)   # [B,1,H,W]
        aw = self.angle_weights.expand(B, -1, -1, -1) # [B, n_angles, H, W]
        n_radii = self._cached_n_radii

        # 确保 weights 在正确设备
        if hasattr(self, 'weights') and self.weights.device != device:
            if isinstance(self.weights, nn.Parameter):
                self.weights.data = self.weights.data.to(device)
            else:
                self.weights = self.weights.to(device)

        gain = torch.zeros(B, self.c_mid, H, W, device=device)
        for c in range(self.c_mid):
            w_c = self.weights[c]  # [n_angles, n_radii]
            gain_c = torch.zeros(B, H, W, device=device)
            for r in range(n_radii):
                mask_r = (r_idx == r).float()  # [B,1,H,W]
                w_r = w_c[:, r]               # [n_angles]
                weighted_angles = aw * w_r.view(1, -1, 1, 1)
                angle_sum = weighted_angles.sum(dim=1)
                gain_c += angle_sum * mask_r.squeeze(1)
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
                 afe_cfg: dict = dict(
                     c_mid=16, n_angles=8, radius_width=8, overlap_ratio=1.5,
                     learnable_weights=True, residual=True, use_hann_window=False),
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        # 侧向层的数量（lateral_convs）由父类根据 start_level 和 in_channels 计算得出
        num_lateral = len(self.lateral_convs)
        self.afe_modules = nn.ModuleList()
        for i in range(num_lateral):
            if enhance_levels is not None and i not in enhance_levels:
                self.afe_modules.append(nn.Identity())
            else:
                self.afe_modules.append(AngleFreqEnhance(in_channels=out_channels, **afe_cfg))

    @auto_fp16()
    def forward(self, inputs):
        # 侧向连接 + 频域增强
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            feat = lateral_conv(inputs[i + self.start_level])
            feat = self.afe_modules[i](feat)
            laterals.append(feat)

        # 自顶向下融合（与原始 FPN 相同）
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

        # 确保输出层数与 num_outs 严格一致（防御性代码）
        assert len(outs) == self.num_outs, \
            f"FPN outputs {len(outs)} levels but num_outs={self.num_outs}. " \
            f"Check add_extra_convs, num_outs and backbone levels."
        return tuple(outs)