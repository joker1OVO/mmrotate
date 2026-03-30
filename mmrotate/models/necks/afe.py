# mmrotate/models/necks/angle_freq_enhance_fpn.py
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
    整张特征图的频域角度高频增强模块（带通道压缩）。

    Args:
        n_angles (int): 角度扇区数量，默认 8。
        high_freq_ratio (float): 高频阈值（半径比例），默认 0.3。
        learnable_weights (bool): 是否学习角度权重，默认 True。
        enhance_init (float): 初始增强系数，默认 1.0。
        residual (bool): 是否使用残差连接，默认 True。
        c_mid (int): 中间通道数（压缩后的通道数），默认 16。
        eps (float): 数值稳定项，默认 1e-8。
    """
    def __init__(self,
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

        # 通道压缩与恢复（假设输入通道为 256，即 FPN 输出通道）
        self.proj_in = nn.Conv2d(256, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, 256, kernel_size=1, bias=False)

        if learnable_weights:
            self.angle_weights = nn.Parameter(torch.full((n_angles,), enhance_init))
        else:
            self.register_buffer('angle_weights', torch.full((n_angles,), enhance_init))

        # 预计算掩码的占位符
        self.register_buffer('angle_idx', None)
        self.register_buffer('high_freq_mask', None)

    def _set_masks(self, H: int, W: int, device: torch.device):
        """生成或更新角度索引和高频掩码（仅当分辨率改变时）"""
        if self.angle_idx is None or self.angle_idx.shape[-2:] != (H, W):
            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(torch.arange(H, device=device),
                                  torch.arange(W, device=device), indexing='ij')
            r = torch.sqrt((y - cy)**2 + (x - cx)**2)
            theta = torch.atan2(y - cy, x - cx) + math.pi  # [0, 2π)

            # 角度扇区索引
            angle_step = 2 * math.pi / self.n_angles
            angle_idx = (theta / angle_step).floor().long() % self.n_angles

            # 高频掩码（半径大于阈值）
            r_max = min(cy, cx)
            high_freq_mask = (r > self.high_freq_ratio * r_max)

            # 存储为 [1, 1, H, W]
            self.angle_idx = angle_idx.unsqueeze(0).unsqueeze(0)
            self.high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0)

    @auto_fp16()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # 1. 通道压缩
        x_proj = self.proj_in(x)  # [B, c_mid, H, W]

        # 2. 获取掩码（按当前分辨率）
        self._set_masks(H, W, device)
        angle_idx = self.angle_idx.expand(B, self.c_mid, -1, -1)      # [B, c_mid, H, W]
        high_freq_mask = self.high_freq_mask.expand(B, self.c_mid, -1, -1)

        # 3. FFT
        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs()
        phase = x_fft_shift.angle()

        # 4. 增益权重
        weights = self.angle_weights[angle_idx]          # [B, c_mid, H, W]
        gain = torch.where(high_freq_mask, weights, torch.ones_like(weights))

        # 5. 应用增益
        mag_enhanced = mag * gain

        # 6. IFFT
        x_fft_enhanced = torch.complex(mag_enhanced, torch.zeros_like(mag_enhanced)) * torch.exp(1j * phase)
        x_fft_ishift = torch.fft.ifftshift(x_fft_enhanced, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real

        # 7. 恢复通道
        x_enh = self.proj_out(x_enh)   # [B, C, H, W]

        # 8. 残差连接
        if self.residual:
            return x + x_enh
        else:
            return x_enh


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """
    支持在指定输出层应用频域角度高频增强的 FPN。

    参数:
        enhance_layers (list[int]): 要增强的 FPN 输出层索引（从 0 开始），
                                    例如 [0] 表示只增强 P2。
        enhance_configs (dict or list[dict]): 每个增强层的配置，若为 dict 则所有层共享。
        ... (其他 FPN 参数)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 enhance_layers: List[int],
                 enhance_configs: Optional[dict] = None,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg)

        self.enhance_layers = enhance_layers

        # 构建增强模块
        if enhance_configs is None:
            enhance_configs = dict()
        if isinstance(enhance_configs, dict):
            self.enhance_modules = nn.ModuleList([
                AngleFreqEnhance(**enhance_configs) for _ in enhance_layers
            ])
        elif isinstance(enhance_configs, list):
            assert len(enhance_configs) == len(enhance_layers), \
                "enhance_configs length must match enhance_layers length"
            self.enhance_modules = nn.ModuleList([
                AngleFreqEnhance(**cfg) for cfg in enhance_configs
            ])
        else:
            raise TypeError("enhance_configs must be dict or list of dict")

    @auto_fp16()
    def forward(self, inputs):
        # 先调用父类前向，得到原始 FPN 输出列表
        outs = super().forward(inputs)  # tuple of Tensors

        # 对指定层应用增强
        outs = list(outs)
        for idx, layer_idx in enumerate(self.enhance_layers):
            if layer_idx < len(outs):
                outs[layer_idx] = self.enhance_modules[idx](outs[layer_idx])
        return tuple(outs)