import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

class FreqResidualModulation(nn.Module):
    """
    频域残差调制模块（可放在FPN融合中）
    对输入特征图进行 FFT，预测幅度谱的残差 ΔM，然后 IFFT 恢复。
    输出 = 输入 + 调制后的特征（空间域残差连接）
    """
    def __init__(self,
                 in_channels: int = 256,
                 c_mid: int = 16,           # 中间通道压缩数
                 kernel_size: int = 3,       # 残差预测器的卷积核大小
                 use_tanh: bool = True):     # 是否用 Tanh 限制 ΔM 范围
        super().__init__()
        self.in_channels = in_channels
        self.c_mid = c_mid
        self.use_tanh = use_tanh

        # 通道压缩与恢复
        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)

        # 残差预测器：一个小型 CNN，输入幅度谱，输出 ΔM
        # 使用两个卷积层 + 激活，保持空间尺寸不变
        padding = kernel_size // 2
        self.res_predictor = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_mid, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(c_mid),
        )
        if use_tanh:
            self.res_predictor.add_module('tanh', nn.Tanh())   # 输出范围 [-1,1]

    @auto_fp16()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] 输入特征图（通常是 FPN 中上采样后的高层特征）
        返回: [B, C, H, W] 增强后的特征图（与 x 尺寸相同）
        """
        B, C, H, W = x.shape
        device = x.device

        # 1. 通道压缩
        x_proj = self.proj_in(x)          # [B, c_mid, H, W]

        # 2. FFT 并中心化
        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs()           # 幅度谱
        phase = x_fft_shift.angle()       # 相位谱

        # 3. 预测残差幅度谱 ΔM
        delta_mag = self.res_predictor(mag)   # [B, c_mid, H, W]

        # 4. 幅度谱残差加法
        mag_enhanced = mag + delta_mag        # 可正可负，允许增强或抑制

        # 5. 重构复数频谱
        x_fft_shift_enhanced = mag_enhanced * torch.exp(1j * phase)

        # 6. 逆 FFT 回空间域
        x_fft_ishift = torch.fft.ifftshift(x_fft_shift_enhanced, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real

        # 7. 恢复通道
        x_enh = self.proj_out(x_enh)

        # 8. 空间域残差连接（保留原始特征）
        return x + x_enh


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 enhance_levels: List[int] = [0,1,2,3],
                 afe_cfg: dict = dict(c_mid=16, kernel_size=3, use_tanh=True),
                 **kwargs):
        super().__init__(in_channels, out_channels, num_outs, **kwargs)
        self.enhance_levels = enhance_levels
        # 为每个层级构建独立的频域残差调制模块（只用在融合时）
        self.freq_mods = nn.ModuleList()
        for i in range(len(in_channels)):
            if i in enhance_levels:
                self.freq_mods.append(
                    FreqResidualModulation(in_channels=out_channels, **afe_cfg))
            else:
                self.freq_mods.append(nn.Identity())

    @auto_fp16()
    def forward(self, inputs):
        # 1. 侧向连接（不做调制，只做通道对齐）
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            feat = lateral_conv(inputs[i + self.start_level])
            laterals.append(feat)   # 注意：这里不再直接应用调制

        # 2. Top-down 融合，在相加前对高层上采样特征做频域残差调制
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
            # 关键：对高层上采样特征进行频域残差调制
            upsampled = self.freq_mods[i](upsampled)   # 这里 i 对应高层层级
            laterals[i-1] = laterals[i-1] + upsampled

        # 3. 后续输出层卷积等保持不变
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        # ... 额外层处理（与之前相同）
        return tuple(outs)