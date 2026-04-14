import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List

class FreqResidualModulation(nn.Module):
    """频域残差调制模块（稳健版本）"""
    def __init__(self,
                 in_channels: int = 256,
                 c_mid: int = 16,
                 kernel_size: int = 3,
                 use_tanh: bool = True,
                 scale: float = 0.05,      # 默认降低到 0.05
                 eps: float = 1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.c_mid = c_mid
        self.use_tanh = use_tanh
        self.scale = scale
        self.eps = eps

        # 通道压缩与恢复（使用 bias=True 增加稳定性）
        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=True)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=True)

        padding = kernel_size // 2
        # 使用 InstanceNorm 替代 BatchNorm，避免统计量不稳定
        self.res_predictor = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm2d(c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_mid, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm2d(c_mid),
        )
        if use_tanh:
            self.res_predictor.add_module('tanh', nn.Tanh())

        # 可选：添加一个可学习的全局缩放因子
        self.global_scale = nn.Parameter(torch.ones(1) * scale, requires_grad=True)

    @auto_fp16()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入检查（调试用，训练稳定后可删除）
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: input contains NaN/Inf")
            return x

        B, C, H, W = x.shape
        # 确保 H, W 为偶数（FFT 更稳定），若为奇数则 pad
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W%2, 0, H%2))

        # 1. 通道压缩
        x_proj = self.proj_in(x)

        # 2. FFT（使用 real FFT 可能更稳定，但保持复数）
        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs()
        phase = x_fft_shift.angle()

        # 3. 预测残差幅度谱
        delta_mag = self.res_predictor(mag)
        if self.use_tanh:
            delta_mag = delta_mag * self.global_scale  # 使用可学习缩放
        else:
            delta_mag = delta_mag * self.scale

        # 4. 幅度增强并 clamp
        mag_enhanced = mag + delta_mag
        mag_enhanced = torch.clamp(mag_enhanced, min=self.eps)

        # 5. 重建复数频谱（使用复数乘法，更稳定）
        # 方法：将幅度和相位合并为复数
        x_fft_shift_enhanced = mag_enhanced * torch.exp(1j * phase)

        # 6. 逆 FFT
        x_fft_ishift = torch.fft.ifftshift(x_fft_shift_enhanced, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real

        # 7. 恢复通道
        x_enh = self.proj_out(x_enh)

        # 8. 残差连接（加入一个可学习的残差权重）
        # 初始时让 residual 权重很小，逐渐增加
        return x + x_enh


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 enhance_levels: List[int] = [1, 2, 3],   # 建议从1开始，不调P2
                 afe_cfg: dict = dict(c_mid=16, kernel_size=3, use_tanh=True, scale=0.05),
                 **kwargs):
        super().__init__(in_channels, out_channels, num_outs, **kwargs)
        self.enhance_levels = enhance_levels
        self.freq_mods = nn.ModuleList()
        for i in range(len(in_channels)):
            if i in enhance_levels:
                self.freq_mods.append(
                    FreqResidualModulation(in_channels=out_channels, **afe_cfg))
            else:
                self.freq_mods.append(nn.Identity())

    @auto_fp16()
    def forward(self, inputs):
        # 1. 侧向连接
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            feat = lateral_conv(inputs[i + self.start_level])
            # 检查侧向卷积后的输出
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                print(f"NaN/Inf detected in lateral_conv output for level {i}")
            laterals.append(feat)

        # 2. Top-down 融合
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
            backbone_idx = i + self.start_level
            if backbone_idx in self.enhance_levels:
                # 检查进入频域模块前的 upsampled
                if torch.isnan(upsampled).any() or torch.isinf(upsampled).any():
                    print(f"NaN/Inf detected before freq_mod for level {backbone_idx}")
                upsampled = self.freq_mods[backbone_idx](upsampled)
                # 检查频域模块输出
                if torch.isnan(upsampled).any() or torch.isinf(upsampled).any():
                    print(f"NaN/Inf detected after freq_mod for level {backbone_idx}")
            laterals[i - 1] = laterals[i - 1] + upsampled
            # 检查相加后的结果
            if torch.isnan(laterals[i - 1]).any() or torch.isinf(laterals[i - 1]).any():
                print(f"NaN/Inf detected after addition for level {i - 1}")

        # 3. 构建输出
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        # 检查最终输出
        for idx, out in enumerate(outs):
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"NaN/Inf detected in final out level {idx}")

        # 处理 extra convs (P6等)
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