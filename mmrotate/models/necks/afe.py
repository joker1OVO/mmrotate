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
    简化版频域角度增强：仅将幅度谱按角度分为 n_angles 个扇区，
    每个扇区一个可学习权重（初始=1），应用到整个幅度谱（包括全部频率）。
    输入特征图 → 1x1 投影到 c_mid → FFT → 幅度谱按角度加权 → IFFT → 1x1 恢复通道 → 残差连接
    """

    def __init__(self,
                 in_channels: int = 256,
                 c_mid: int = 16,
                 n_angles: int = 8,
                 learnable_weights: bool = True,
                 residual: bool = True,
                 eps: float = 1e-8):
        super().__init__()
        self.c_mid = c_mid
        self.n_angles = n_angles
        self.residual = residual
        self.eps = eps

        # 投影层
        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj_in.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.proj_out.weight, mode='fan_out', nonlinearity='relu')

        # 每个角度扇区一个可学习权重（形状: (c_mid, n_angles)）
        if learnable_weights:
            self.angle_weights = nn.Parameter(torch.ones(c_mid, n_angles))
        else:
            self.register_buffer('angle_weights', torch.ones(c_mid, n_angles))

        # 缓存角度索引掩码（与特征图尺寸相关）
        self.register_buffer('angle_idx', None)
        self._cached_HW = None

    def _build_angle_mask(self, H: int, W: int, device: torch.device):
        """生成每个像素的角度索引（0 ~ n_angles-1）"""
        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing='ij')
        theta = torch.atan2(y - cy, x - cx) + math.pi   # [0, 2π)
        # 将角度映射到 0 ~ n_angles-1（均匀扇区）
        angle_idx = (theta / (2 * math.pi / self.n_angles)).floor().long() % self.n_angles
        self.angle_idx = angle_idx.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        self._cached_HW = (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # 投影到低维通道
        x_proj = self.proj_in(x)   # [B, c_mid, H, W]

        # FFT
        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs() + self.eps

        # 构建角度掩码（若尺寸变化则重新生成）
        if self._cached_HW != (H, W):
            self._build_angle_mask(H, W, device)

        # angle_idx: [1,1,H,W] -> 扩展到 batch 维度 [B,1,H,W]
        angle_idx = self.angle_idx.expand(B, -1, H, W)   # [B,1,H,W]

        # 计算增益 [B, c_mid, H, W]
        gain = torch.zeros(B, self.c_mid, H, W, device=device)
        for c in range(self.c_mid):
            # w: [n_angles]
            w = self.angle_weights[c]
            # 使用高级索引：w[angle_idx] 得到形状 [B,1,H,W]，然后移除 channel 维度
            gain_c = w[angle_idx]      # [B,1,H,W]
            gain[:, c, :, :] = gain_c.squeeze(1)

        # 应用增益到幅度谱
        mag_enhanced = mag * gain

        # 相位保持不变，重构复数谱并逆变换
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
    """在 FPN 侧向连接后添加频域角度增强（简化版）"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 enhance_levels: Optional[List[int]] = None,
                 afe_cfg: dict = dict(c_mid=16, n_angles=8, learnable_weights=True, residual=True),
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
        # 侧向连接 + AFE
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

        # 输出卷积
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # 额外层（P6, P7 ...）
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                # 如果使用 extra convs，简化处理：保持原逻辑但保持 num_outs 一致
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