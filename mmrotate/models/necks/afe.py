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
    频域极坐标增强模块：
    - 仅对高频区域（r > high_freq_ratio * max_r）进行半径+角度分组增强
    - 每个(角度扇区, 半径环)对应一个可学习权重
    - 所有通道共享同一组权重
    - 无重叠，硬分配
    """

    def __init__(self,
                 in_channels: int = 256,
                 n_angles: int = 8,
                 n_radii: int = 4,          # 半径分组数
                 radius_width: int = 8,     # 均匀半径宽度（如果 n_radii 不指定则用此）
                 high_freq_ratio: float = 0.3,
                 learnable_weights: bool = True,
                 enhance_init: float = 1.0,
                 residual: bool = True,
                 c_mid: int = 16,
                 eps: float = 1e-8):
        super().__init__()
        self.n_angles = n_angles
        self.n_radii = n_radii
        self.radius_width = radius_width
        self.high_freq_ratio = high_freq_ratio
        self.residual = residual
        self.c_mid = c_mid
        self.eps = eps

        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)

        # 权重形状 (n_angles, n_radii)
        if learnable_weights:
            self.weights = nn.Parameter(torch.full((n_angles, n_radii), enhance_init))
        else:
            self.register_buffer('weights', torch.full((n_angles, n_radii), enhance_init))

        # 缓存掩码
        self.register_buffer('radius_idx', None)
        self.register_buffer('angle_idx', None)
        self.register_buffer('high_freq_mask', None)
        self._cached_HW = None
        self._cached_n_radii = None

    def _build_masks(self, H: int, W: int, device: torch.device):
        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing='ij')
        r = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        theta = torch.atan2(y - cy, x - cx) + math.pi   # [0, 2π)

        # 半径分区：均匀划分
        max_r = max(cy, cx)
        if self.n_radii is not None:
            # 使用固定数量分区
            n_radii = self.n_radii
            radius_idx = torch.floor(r / (max_r / n_radii)).long().clamp(0, n_radii - 1)
        else:
            # 使用固定宽度
            n_radii = int(max_r // self.radius_width) + 1
            radius_idx = torch.floor(r / self.radius_width).long().clamp(0, n_radii - 1)

        # 角度分区：均匀划分，无重叠
        angle_step = 2 * math.pi / self.n_angles
        angle_idx = (theta / angle_step).floor().long() % self.n_angles

        # 高频掩码
        high_freq_mask = (r > self.high_freq_ratio * max_r)

        self.radius_idx = radius_idx.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        self.angle_idx = angle_idx.unsqueeze(0).unsqueeze(0)     # [1,1,H,W]
        self.high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0)
        self._cached_HW = (H, W)
        self._cached_n_radii = n_radii

        # 如果权重参数与当前 n_radii 不符，动态调整（通常不应发生）
        if hasattr(self, 'weights') and self.weights is not None:
            if self.weights.size(1) != n_radii:
                new_weights = torch.full((self.n_angles, n_radii), 1.0, device=device)
                if isinstance(self.weights, nn.Parameter):
                    self.weights = nn.Parameter(new_weights)
                else:
                    self.weights = new_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        x_proj = self.proj_in(x)   # [B, c_mid, H, W]

        if self._cached_HW != (H, W):
            self._build_masks(H, W, device)

        # 扩展掩码到 batch 维度（通道维度后续广播）
        radius_idx = self.radius_idx.expand(B, -1, H, W)   # [B,1,H,W]
        angle_idx = self.angle_idx.expand(B, -1, H, W)     # [B,1,H,W]
        hf_mask = self.high_freq_mask.expand(B, -1, H, W)  # [B,1,H,W]

        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs() + self.eps

        # 计算增益：基于 radius_idx 和 angle_idx 从 weights 中索引
        # weights: [n_angles, n_radii]
        # 生成形状 [B,1,H,W] 的增益矩阵
        # 使用高级索引
        gain = torch.ones(B, 1, H, W, device=device)
        # 对于每个半径组，获取对应角度的权重，并赋给对应位置
        for r in range(self._cached_n_radii):
            mask_r = (radius_idx == r)   # [B,1,H,W]
            if not mask_r.any():
                continue
            # 获取该半径环对应的角度权重向量（形状 [n_angles]）
            w_r = self.weights[:, r]     # [n_angles]
            # 对于每个角度，从 w_r 中取值
            # 使用 angle_idx 作为索引：angle_idx [B,1,H,W] 取值范围 0..n_angles-1
            # 通过 gather 或直接 w_r[angle_idx]
            w_per_position = w_r[angle_idx]   # [B,1,H,W] (因为 angle_idx 是 (B,1,H,W))
            gain = torch.where(mask_r, w_per_position, gain)

        # 只在高频区域应用增益，低频区域保持1
        gain = torch.where(hf_mask, gain, torch.ones_like(gain))

        # 将增益扩展到 c_mid 通道（所有通道使用相同增益）
        gain = gain.expand(B, self.c_mid, H, W)   # [B, c_mid, H, W]

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
                     n_angles=8, n_radii=4, radius_width=8, high_freq_ratio=0.3,
                     learnable_weights=True, enhance_init=1.0, residual=True, c_mid=16),
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