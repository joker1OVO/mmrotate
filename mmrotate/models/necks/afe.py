import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS


class AFEModule(nn.Module):
    """
    单输入方向频率增强模块 (AFE)。
    作用：对单层特征图进行频域主轴提取并进行各向异性高斯滤波增强。
    """

    def __init__(self, channels, m=7, eps=1e-8):
        super(AFEModule, self).__init__()
        self.m = m
        self.channels = channels
        self.eps = eps

        self.param_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(m * m, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        self.refine_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def generate_anisotropic_kernel(self, params, device):
        B = params.shape[0]
        theta = torch.atan2(params[:, 0], params[:, 1]) / 2.0 + (math.pi / 2.0)
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

    def forward(self, x):
        B, C, H, W = x.shape
        m = self.m
        x_f = torch.fft.fft2(x, norm='ortho')
        x_f = torch.fft.fftshift(x_f, dim=(-2, -1))

        mag_center = torch.abs(x_f).mean(dim=1, keepdim=True)[
            :, :, H // 2 - m // 2: H // 2 + m // 2 + 1, W // 2 - m // 2: W // 2 + m // 2 + 1]
        params = self.param_net(mag_center)

        kernel = self.generate_anisotropic_kernel(params, x.device).repeat_interleave(C, dim=0).unsqueeze(1)
        real_f = F.conv2d(x_f.real.view(1, B * C, H, W), kernel, padding=m // 2, groups=B * C).view(B, C, H, W)
        imag_f = F.conv2d(x_f.imag.view(1, B * C, H, W), kernel, padding=m // 2, groups=B * C).view(B, C, H, W)

        out_f = torch.complex(self.refine_conv(real_f), self.refine_conv(imag_f))
        out_f = torch.fft.ifftshift(out_f, dim=(-2, -1))
        return torch.fft.ifft2(out_f, norm='ortho').real + x


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 enhance_levels=[0, 1, 2, 3],
                 fam_cfg=dict(m=7, eps=1e-8),
                 **kwargs):
        # 显式移除 fusion_modes 避免传给父类 FPN
        kwargs.pop('fusion_modes', None)
        super(AngleFreqEnhanceFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        self.enhance_levels = enhance_levels
        self.afe_modules = nn.ModuleList()
        for i in range(len(in_channels)):
            if i in enhance_levels:
                self.afe_modules.append(AFEModule(out_channels, **fam_cfg))
            else:
                self.afe_modules.append(nn.Identity())

    @auto_fp16()
    def forward(self, inputs):
        # 1. 横向连接
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 2. 对特定层进行 AFE 增强 (ResNetStage -> LateralConv -> AFE)
        for i in range(len(laterals)):
            if i in self.enhance_levels:
                laterals[i] = self.afe_modules[i](laterals[i])

        # 3. 自上而下路径融合 (Top-down)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)

        # 4. 构建输出 P2-P5
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # 5. 修正的关键：生成额外的特征层 (P6/P7)
        # 必须确保生成的 outs 长度等于 num_outs (通常是 5)
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