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
    基于 afe7595 添加半径分组功能。
    - 保留原有角度权重和高频掩码。
    - 新增：在半径方向划分圆环，每个圆环一个可学习权重（仅在有效高频区域应用）。
    - 开关 use_radius_groups 控制是否启用半径分组；关闭时行为与原版一致。
    """

    def __init__(self,
                 in_channels: int = 256,
                 n_angles: int = 8,
                 high_freq_ratio: float = 0.3,
                 learnable_weights: bool = True,
                 enhance_init: float = 1.0,
                 residual: bool = True,
                 c_mid: int = 16,
                 eps: float = 1e-8,
                 use_radius_groups: bool = False,      # 新增：半径分组开关
                 radius_width: int = 8,                # 半径分组宽度（像素）
                 n_radii: int = 8,                     # 半径分组数量（若指定则覆盖 radius_width）
                 ):
        super().__init__()
        self.n_angles = n_angles
        self.high_freq_ratio = high_freq_ratio
        self.residual = residual
        self.c_mid = c_mid
        self.eps = eps
        self.use_radius_groups = use_radius_groups

        # 动态投影层
        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)

        # 角度权重（与原版一致）
        if learnable_weights:
            self.angle_weights = nn.Parameter(torch.full((n_angles,), enhance_init))
        else:
            self.register_buffer('angle_weights', torch.full((n_angles,), enhance_init))

        # 半径分组权重（仅当 use_radius_groups=True 时使用）
        if use_radius_groups:
            # 半径分组数量（如果指定 n_radii 则用它，否则根据 radius_width 和最大半径计算）
            self.register_buffer('_radius_width', torch.tensor(radius_width))
            # 半径权重：形状 (n_radii, )，初始为1
            if learnable_weights:
                self.radius_weights = nn.Parameter(torch.ones(n_radii))
            else:
                self.register_buffer('radius_weights', torch.ones(n_radii))
        else:
            self.radius_weights = None

        # 缓冲区
        self.register_buffer('angle_idx', None)
        self.register_buffer('high_freq_mask', None)
        self.register_buffer('radius_idx', None)   # 半径索引（仅当 use_radius_groups=True）

    def _set_masks(self, H: int, W: int, device: torch.device):
        """生成角度索引、高频掩码，以及可选的半径索引"""
        if self.angle_idx is None or self.angle_idx.shape[-2:] != (H, W):
            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(torch.arange(H, device=device),
                                  torch.arange(W, device=device), indexing='ij')
            r = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            theta = torch.atan2(y - cy, x - cx) + math.pi

            # 角度索引（与原版一致）
            angle_step = 2 * math.pi / self.n_angles
            angle_idx = (theta / angle_step).floor().long() % self.n_angles

            # 高频掩码
            r_max = max(cy, cx) if max(cy, cx) > 0 else 1
            high_freq_mask = (r > self.high_freq_ratio * r_max)

            self.angle_idx = angle_idx.unsqueeze(0).unsqueeze(0)
            self.high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0)

            # 半径索引（如果需要）
            if self.use_radius_groups:
                # 计算半径分组数
                if hasattr(self, '_n_radii'):
                    n_radii = self._n_radii
                else:
                    # 根据 radius_width 计算
                    max_r = r_max
                    n_radii = int(max_r // self._radius_width.item()) + 1
                    self._n_radii = n_radii
                radius_idx = torch.floor(r / self._radius_width.item()).long().clamp(0, n_radii - 1)
                self.radius_idx = radius_idx.unsqueeze(0).unsqueeze(0)

                # 调整 radius_weights 的维度
                if self.radius_weights is not None and self.radius_weights.size(0) != n_radii:
                    new_weights = torch.ones(n_radii, device=device)
                    if isinstance(self.radius_weights, nn.Parameter):
                        self.radius_weights = nn.Parameter(new_weights)
                    else:
                        self.radius_weights = new_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_proj = self.proj_in(x)   # [B, c_mid, H, W]

        self._set_masks(H, W, x.device)

        # 扩展掩码到 batch 和 c_mid 通道
        a_idx = self.angle_idx.expand(B, self.c_mid, -1, -1)
        hf_mask = self.high_freq_mask.expand(B, self.c_mid, -1, -1)

        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs() + self.eps

        # 基础角度权重
        weights_angle = self.angle_weights[a_idx]   # [B, c_mid, H, W]

        # 半径分组权重（如果启用）
        if self.use_radius_groups and self.radius_idx is not None:
            r_idx = self.radius_idx.expand(B, self.c_mid, -1, -1)   # [B, c_mid, H, W]
            # 为每个空间位置分配半径权重
            # radius_weights 是一维 [n_radii]
            if self.radius_weights is not None:
                radius_gain = self.radius_weights[r_idx]            # [B, c_mid, H, W]
                # 最终增益 = 角度权重 * 半径权重
                gain = weights_angle * radius_gain
            else:
                gain = weights_angle
        else:
            gain = weights_angle

        # 只在高频区域应用增益，低频区域增益为1
        gain = torch.where(hf_mask, gain, torch.ones_like(gain))

        x_fft_shift = x_fft_shift * gain

        x_fft_ishift = torch.fft.ifftshift(x_fft_shift, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real
        x_enh = self.proj_out(x_enh)

        return x + x_enh if self.residual else x_enh


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """与 afe7595 中相同的 FPN 封装（仅添加半径分组参数）"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 enhance_levels: List[int] = [0, 1, 2, 3],
                 afe_cfg: dict = dict(n_angles=8, c_mid=16),
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
            # 处理额外层（与原始 FPN 一致）
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