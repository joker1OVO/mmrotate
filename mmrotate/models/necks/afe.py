import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List

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
                 use_tanh: bool = True,      # 是否用 Tanh 限制 ΔM 范围
                 scale: float = 0.1):        # 残差缩放因子，控制修改幅度
        super().__init__()
        self.in_channels = in_channels
        self.c_mid = c_mid
        self.use_tanh = use_tanh
        self.scale = scale

        # 通道压缩与恢复
        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)

        # 残差预测器：一个小型 CNN，输入幅度谱，输出 ΔM
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
        x: [B, C, H, W] 输入特征图
        返回: [B, C, H, W] 增强后的特征图
        """
        B, C, H, W = x.shape
        device = x.device

        # 1. 通道压缩
        x_proj = self.proj_in(x)          # [B, c_mid, H, W]

        # 2. FFT 并中心化
        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs()           # 幅度谱，非负
        phase = x_fft_shift.angle()       # 相位谱

        # 3. 预测残差幅度谱 ΔM，并应用缩放因子
        delta_mag = self.res_predictor(mag) * self.scale   # [B, c_mid, H, W]

        # 4. 幅度谱残差加法（允许正负），并保证非负
        mag_enhanced = mag + delta_mag
        mag_enhanced = torch.clamp(mag_enhanced, min=1e-6)  # 防止后续 log 或除零

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
                 enhance_levels: List[int] = [0, 1, 2, 3],
                 afe_cfg: dict = dict(c_mid=16, kernel_size=3, use_tanh=True, scale=0.1),
                 **kwargs):
        super().__init__(in_channels, out_channels, num_outs, **kwargs)
        self.enhance_levels = enhance_levels
        # 注意：backbone 输出的特征层数为 len(in_channels)
        # 这里为每个 backbone 输出层都构建一个调制器（未在 enhance_levels 中的使用 Identity）
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
            laterals.append(feat)

        # 2. Top-down 融合，在相加前对高层上采样特征做频域残差调制
        used_backbone_levels = len(laterals)
        # 注意：laterals[i] 对应 backbone 的 i + self.start_level 层
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
            # 关键修正：根据 backbone 的实际层级索引来获取对应的调制器
            backbone_idx = i + self.start_level
            upsampled = self.freq_mods[backbone_idx](upsampled)
            laterals[i-1] = laterals[i-1] + upsampled

        # 3. 构建输出层（P2~P5 等）
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # 4. 处理额外的输出层（如 P6, P7）—— 保留父类的额外层逻辑
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