import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS


class AFEModule(nn.Module):
    """
    AngleFreqEnhance (AFE) Module - 方案二实现
    利用 FFT 幅度谱生成动态各向异性高斯卷积核，增强旋转目标特征。
    """

    def __init__(self, channels, m=7, eps=1e-8):
        super(AFEModule, self).__init__()
        self.m = m
        self.channels = channels
        self.eps = eps

        # 提取参数 theta, lambda1, lambda2 的轻量级网络
        # 连续性保护：回归 sin(2theta), cos(2theta) 以及 log(lambda)
        self.param_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(m * m, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        # 实部与虚部共享的 1x1 信息整合卷积
        self.refine_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def generate_anisotropic_kernel(self, params, device):
        """
        根据回归参数生成 m*m 的各向异性高斯核
        """
        B = params.shape[0]

        # 1. 连续性保护：处理 sin/cos 预测值
        theta = torch.atan2(params[:, 0], params[:, 1]) / 2.0

        # 2. 角度修正：补偿 90 度以对齐物理主轴
        theta = theta + (math.pi / 2.0)

        # 3. 尺度保护：使用 exp 确保尺度为正
        lam1 = torch.exp(params[:, 2])
        lam2 = 1.0 / (lam1 + self.eps)

        # 4. 生成坐标网格
        y, x = torch.meshgrid(
            torch.linspace(-(self.m // 2), self.m // 2, self.m, device=device),
            torch.linspace(-(self.m // 2), self.m // 2, self.m, device=device)
        )
        x = x.repeat(B, 1, 1)
        y = y.repeat(B, 1, 1)

        # 5. 旋转映射
        cos_t, sin_t = torch.cos(theta).view(B, 1, 1), torch.sin(theta).view(B, 1, 1)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t

        # 6. 计算各向异性高斯分布权重
        kernel = torch.exp(
            -((x_rot ** 2) / (2 * lam1.view(B, 1, 1) ** 2) + (y_rot ** 2) / (2 * lam2.view(B, 1, 1) ** 2)))

        # 7. 归一化
        kernel_sum = kernel.sum(dim=(1, 2), keepdim=True) + self.eps
        kernel = kernel / kernel_sum
        return kernel

    def forward(self, x_high, x_low):
        B, C, H, W = x_low.shape
        m = self.m

        # 上采样高层特征
        x_high_up = F.interpolate(x_high, size=(H, W), mode='bilinear', align_corners=False)

        # --- 步骤 1: FFT 变换 ---
        x_f = torch.fft.fft2(x_low, norm='ortho')
        x_f = torch.fft.fftshift(x_f, dim=(-2, -1))
        real, imag = x_f.real, x_f.imag

        # --- 步骤 2: 方向参数提取 ---
        mag = torch.abs(x_f)
        mag_weighted = torch.mean(mag, dim=1, keepdim=True)

        h_idx, w_idx = H // 2, W // 2
        mag_center = mag_weighted[:, :, h_idx - m // 2: h_idx + m // 2 + 1, w_idx - m // 2: w_idx + m // 2 + 1]

        params = self.param_net(mag_center)

        # --- 步骤 3: 动态卷积 ---
        dynamic_kernel = self.generate_anisotropic_kernel(params, x_low.device)
        dynamic_kernel = dynamic_kernel.repeat_interleave(C, dim=0).unsqueeze(1)

        real_f = F.conv2d(real.view(1, B * C, H, W), dynamic_kernel,
                          padding=m // 2, groups=B * C).view(B, C, H, W)
        imag_f = F.conv2d(imag.view(1, B * C, H, W), dynamic_kernel,
                          padding=m // 2, groups=B * C).view(B, C, H, W)

        # --- 步骤 4: 整合与逆变换 ---
        real_f = self.refine_conv(real_f)
        imag_f = self.refine_conv(imag_f)

        out_f = torch.complex(real_f, imag_f)
        out_f = torch.fft.ifftshift(out_f, dim=(-2, -1))
        x_low_enhanced = torch.fft.ifft2(out_f, norm='ortho').real

        return x_low_enhanced + x_high_up


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """
    带有角度频率增强融合策略的 FPN。
    完全模仿 FAAFusionFPN 的参数传递与逻辑结构。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes=['add', 'add', 'afe'],
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                 fam_cfg=dict(m=7, eps=1e-8)):

        # 模仿 faafusion.py 直接调用父类初始化，避开参数名不匹配问题
        super(AngleFreqEnhanceFPN, self).__init__(
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

        # 验证融合模式长度：4层输入需要3次融合
        backbone_levels = self.backbone_end_level - self.start_level
        expected_steps = backbone_levels - 1
        assert len(fusion_modes) == expected_steps, \
            f"fusion_modes 长度需为 {expected_steps}"

        self.fusion_modes = fusion_modes
        self.afe_modules = nn.ModuleList()

        for mode in fusion_modes:
            if mode == 'afe':
                self.afe_modules.append(AFEModule(out_channels, **fam_cfg))
            else:
                self.afe_modules.append(None)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # 1. 建立横向连接
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 2. 自上而下融合
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 映射 i=3->idx=0 (P5->P4), i=2->idx=1 (P4->P3), i=1->idx=2 (P3->P2)
            fusion_idx = used_backbone_levels - 1 - i
            mode = self.fusion_modes[fusion_idx]

            if mode == 'add':
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
            elif mode == 'afe':
                # 调用 AFE 模块进行增强融合
                laterals[i - 1] = self.afe_modules[fusion_idx](laterals[i], laterals[i - 1])

        # 3. 输出卷积
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # 4. 额外层 (P6/P7) 逻辑完全继承自 FPN
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