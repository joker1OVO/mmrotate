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
    频域角度增强模块：将特征图投影到中继通道，进行频域角度权重增强，再还原。
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

        # 动态投影层，适应输入通道
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

            angle_step = 2 * math.pi / self.n_angles
            angle_idx = (theta / angle_step).floor().long() % self.n_angles
            r_max = max(cy, cx) if max(cy, cx) > 0 else 1
            high_freq_mask = (r > self.high_freq_ratio * r_max)

            self.angle_idx = angle_idx.unsqueeze(0).unsqueeze(0)
            self.high_freq_mask = high_freq_mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_proj = self.proj_in(x)

        self._set_masks(H, W, x.device)
        # 扩展掩码以匹配投影后的通道数 c_mid
        a_idx = self.angle_idx.expand(B, self.c_mid, -1, -1)
        hf_mask = self.high_freq_mask.expand(B, self.c_mid, -1, -1)

        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # 应用角度权重
        weights = self.angle_weights[a_idx]
        gain = torch.where(hf_mask, weights, torch.ones_like(weights))

        x_fft_shift = x_fft_shift * gain

        x_fft_ishift = torch.fft.ifftshift(x_fft_shift, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real
        x_enh = self.proj_out(x_enh)

        return x + x_enh if self.residual else x_enh


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """
    创新点：在 FPN 的侧向连接处对 P2-P5 分别进行频域角度增强，随后进行 Top-down 融合。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 # 控制哪些层需要 AFE，默认 [0,1,2,3] 对应 P2, P3, P4, P5
                 enhance_levels: List[int] = [0, 1, 2, 3],
                 afe_cfg: dict = dict(n_angles=32, c_mid=16),
                 **kwargs):
        super(AngleFreqEnhanceFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        self.enhance_levels = enhance_levels

        # 为每个 backbone 级别构建独立的 AFE 模块
        self.afe_modules = nn.ModuleList()
        for i in range(len(in_channels)):
            if i in enhance_levels:
                # 传入 out_channels，因为 AFE 作用在侧向连接（256通道）之后
                self.afe_modules.append(AngleFreqEnhance(in_channels=out_channels, **afe_cfg))
            else:
                self.afe_modules.append(nn.Identity())

    @auto_fp16()
    def forward(self, inputs):
        # 1. 构建侧向连接并立即应用 AFE 增强
        # laterals[0]->P2, laterals[1]->P3, laterals[2]->P4, laterals[3]->P5
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            feat = lateral_conv(inputs[i + self.start_level])
            # 对当前层进行 AFE 增强（P2-P5 独立处理）
            feat = self.afe_modules[i](feat)
            laterals.append(feat)

        # 2. Top-down 融合
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 这里的融合逻辑：上一层上采样 + 当前层(已增强)
            prev_shape = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 3. 构建输出层 (P2-P5 的 3x3 卷积)
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # 4. 额外的层 (如 P6, P7)
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