import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS


class AFEModule(nn.Module):
    """
    AngleFreqEnhance (AFE) Fusion Module
    在自上而下融合路径中起作用，利用 x_low 的频域信息增强特征，并与 x_high 融合。
    """

    def __init__(self, channels, m=7, eps=1e-8):
        super(AFEModule, self).__init__()
        self.m = m
        self.channels = channels
        self.eps = eps

        # 参数回归网络：提取主轴方向和尺度
        self.param_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(m * m, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        # 频域信息整合卷积
        self.refine_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def generate_anisotropic_kernel(self, params, device):
        B = params.shape[0]
        # 角度连续性保护
        theta = torch.atan2(params[:, 0], params[:, 1]) / 2.0 + (math.pi / 2.0)
        # 尺度保护
        lam1 = torch.exp(params[:, 2])
        lam2 = 1.0 / (lam1 + self.eps)

        y, x = torch.meshgrid(
            torch.linspace(-(self.m // 2), self.m // 2, self.m, device=device),
            torch.linspace(-(self.m // 2), self.m // 2, self.m, device=device)
        )
        x = x.repeat(B, 1, 1)
        y = y.repeat(B, 1, 1)

        cos_t, sin_t = torch.cos(theta).view(B, 1, 1), torch.sin(theta).view(B, 1, 1)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t

        kernel = torch.exp(
            -((x_rot ** 2) / (2 * lam1.view(B, 1, 1) ** 2) + (y_rot ** 2) / (2 * lam2.view(B, 1, 1) ** 2)))
        return kernel / (kernel.sum(dim=(1, 2), keepdim=True) + self.eps)

    def forward(self, x_high, x_low):
        B, C, H, W = x_low.shape
        m = self.m

        # 1. 频域处理主体：x_low (256通道图像)
        x_f = torch.fft.fft2(x_low, norm='ortho')
        x_f = torch.fft.fftshift(x_f, dim=(-2, -1))

        # 2. 提取幅度谱主轴参数
        mag_center = torch.abs(x_f).mean(dim=1, keepdim=True)[
            :, :, H // 2 - m // 2: H // 2 + m // 2 + 1, W // 2 - m // 2: W // 2 + m // 2 + 1]
        params = self.param_net(mag_center)

        # 3. 生成并应用动态核
        kernel = self.generate_anisotropic_kernel(params, x_low.device).repeat_interleave(C, dim=0).unsqueeze(1)
        real_f = F.conv2d(x_f.real.view(1, B * C, H, W), kernel, padding=m // 2, groups=B * C).view(B, C, H, W)
        imag_f = F.conv2d(x_f.imag.view(1, B * C, H, W), kernel, padding=m // 2, groups=B * C).view(B, C, H, W)

        # 4. 复数域整合与逆变换
        out_f = torch.complex(self.refine_conv(real_f), self.refine_conv(imag_f))
        out_f = torch.fft.ifftshift(out_f, dim=(-2, -1))
        x_low_enhanced = torch.fft.ifft2(out_f, norm='ortho').real

        # 5. 自上而下融合：上采样高层特征并相加
        x_high_up = F.interpolate(x_high, size=(H, W), mode='bilinear', align_corners=False)
        return x_low_enhanced + x_high_up


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes=['add', 'add', 'afe'],  # 回到 fusion_modes 模式
                 fam_cfg=dict(m=7, eps=1e-8),
                 **kwargs):
        super(AngleFreqEnhanceFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        self.fusion_modes = fusion_modes
        self.afe_modules = nn.ModuleList()

        # 针对 4 层 Backbone (C2-C5)，会有 3 次融合过程
        for mode in fusion_modes:
            if mode == 'afe':
                self.afe_modules.append(AFEModule(out_channels, **fam_cfg))
            else:
                self.afe_modules.append(None)

    @auto_fp16()
    def forward(self, inputs):
        # Step 1: 横向连接
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Step 2: 自上而下路径融合 (核心回退点)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 获取对应的融合索引，P5->P4 为 0, P4->P3 为 1, P3->P2 为 2
            fusion_idx = used_backbone_levels - 1 - i
            mode = self.fusion_modes[fusion_idx]

            if mode == 'afe':
                # AFE 模式：传入 x_high (laterals[i]) 和 x_low (laterals[i-1])
                laterals[i - 1] = self.afe_modules[fusion_idx](laterals[i], laterals[i - 1])
            else:
                # 标准加法模式
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # Step 3: 输出卷积
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # Step 4: 额外层生成 (保证 RPN 不会报层数不对的错误)
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