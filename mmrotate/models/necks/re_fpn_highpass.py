import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import ROTATED_NECKS
from ..SAFF_DETR import MRF, HighPassFilter  # 假设已注册


@ROTATED_NECKS.register_module()
class re_fpn_highpass(BaseModule):
    """Reduced Refined FPN with P2-P3 fusion using MRF and RepC3.

    Now outputs 5 feature maps (P2, P3, P4, P5, P6) to match original FPN.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,  # 改为5
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(re_fpn_highpass, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        # 验证配置
        assert len(in_channels) == 4, "需要4个输入特征层 [C2, C3, C4, C5]"

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        # ==================== 1. 特征降维卷积 ====================
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        # ==================== 2. P5 高通滤波增强 ====================
        self.spatial_attn_p5 = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.highpass_p5 = HighPassFilter(alpha=0.25, trainable_alpha=False)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

        # 可选：MRF 模块（未使用，但保留）
        self.mrf = MRF(dim=out_channels)

        # ==================== 3. 生成 P6 的下采样卷积 ====================
        if self.num_outs > 4:
            self.extra_downsample = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

        # ==================== 4. FPN 输出卷积（5层） ====================
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):  # 改为 num_outs
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # 1. 降维所有特征层
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]  # laterals[0]~[3] 对应 P2~P5

        # 2. P5 高通滤波增强
        p5_orig = laterals[3]
        high_freq = self.highpass_p5(p5_orig)
        spatial_weight = self.spatial_attn_p5(high_freq)
        p5_attended = p5_orig * spatial_weight
        p5_out = p5_orig + self.residual_weight * p5_attended
        laterals[3] = p5_out

        # 3. 自顶向下融合
        used_backbone_levels = len(laterals)  # 4

        # P5 → P4
        if used_backbone_levels > 2:
            prev_shape = laterals[2].shape[2:]
            laterals[2] = laterals[2] + F.interpolate(
                laterals[3], size=prev_shape, **self.upsample_cfg)

        # P4 → P3
        if used_backbone_levels > 1:
            prev_shape = laterals[1].shape[2:]
            laterals[1] = laterals[1] + F.interpolate(
                laterals[2], size=prev_shape, **self.upsample_cfg)

        # P3 → P2
        if used_backbone_levels > 0:
            prev_shape = laterals[0].shape[2:]
            laterals[0] = laterals[0] + F.interpolate(
                laterals[1], size=prev_shape, **self.upsample_cfg)

        # 4. 生成 P6（如果需要）
        if self.num_outs > 4:
            p6 = self.extra_downsample(laterals[3])  # 对 P5 下采样
            laterals.append(p6)  # laterals 现在有 5 层：P2,P3,P4,P5,P6

        # 5. 应用输出卷积
        outs = []
        for i in range(self.num_outs):
            out_feat = self.fpn_convs[i](laterals[i])
            outs.append(out_feat)

        return tuple(outs)