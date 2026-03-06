# mmrotate/models/necks/re_fpn_mrf45.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import ROTATED_NECKS

# 导入依赖模块（假设已经注册）
from ..SAFF_DETR import MRF, RepC3


@ROTATED_NECKS.register_module()
class re_fpn_mrf45(BaseModule):
    """Reduced Refined FPN with P2-P3 fusion using MRF and RepC3.

    Design: FPN that fuses P2 and P3 using MRF module, then outputs P3-P5.
    This provides a fair comparison with SAFFNeck.

    Args:
        in_channels (list[int]): Input channels per scale [256, 512, 1024, 2048]
        out_channels (int): Output channels (fixed to 256)
        num_outs (int): Number of output scales (fixed to 3)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=3,
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
        super(re_fpn_mrf45, self).__init__(init_cfg)
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
        assert num_outs == 3, "为公平对比，输出必须为3层 [P3, P4, P5]"

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

        # ==================== 2. P2-P3融合模块（使用MRF和RepC3） ====================
        # P2下采样到P3分辨率（4× → 8×）
        self.p2_downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 多分支表示融合 (MRF) - 与SAFFNeck保持一致
        self.mrf_p23 = MRF(dim=out_channels * 2)  # 输入是拼接的P2和P3

        # RepC3高效特征提取 - 与SAFFNeck保持一致
        self.rep_c3_p3 = RepC3(
            c1=out_channels * 2,  # P2下采样 + P3
            c2=out_channels,
            n=3,  # 3次重复
            e=0.5
        )

        # ==================== 2. P4-P5频率增强（使用MRF） ====================
        self.mrf_p4 = MRF(dim=out_channels)  # 新增：P4增强
        self.mrf_p5 = MRF(dim=out_channels)  # 新增：P5增强

        # ==================== 3. FPN卷积层（只针对P3-P5） ====================
        self.fpn_convs = nn.ModuleList()
        # 只构建3个FPN卷积层，对应P3, P4, P5
        for i in range(3):
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
        """Forward function with P2-P3 fusion using MRF and RepC3.

        Processing flow:
        1. Reduce dimensions of all input features (C2-C5)
        2. Fuse P2 and P3 using MRF and RepC3
        3. Standard FPN top-down pathway (P5→P4→P3)
        4. Output P3, P4, P5
        """
        assert len(inputs) == len(self.in_channels)

        # 1. 降维所有特征层
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # laterals[0]: P2 (from C2)
        # laterals[1]: P3 (from C3)
        # laterals[2]: P4 (from C4)
        # laterals[3]: P5 (from C5)

        # 2. 使用MRF和RepC3融合P2和P3
        p2 = laterals[0]  # P2特征
        p3_raw = laterals[1]  # P3原始特征
        # 下采样P2到P3分辨率
        p2_down = self.p2_downsample(p2)
        # 拼接P2和P3
        p23_concat = torch.cat([p2_down, p3_raw], dim=1)
        # MRF多分支表示融合
        p23_mrf = self.mrf_p23(p23_concat)
        # RepC3高效特征提取
        p3_enhanced = self.rep_c3_p3(p23_mrf)
        # 更新laterals，用融合后的P3替换原始的P3
        laterals[1] = p3_enhanced

        # 3. 频率增强P4，P5
        laterals[2] = self.mrf_p4(laterals[2])  # 增强P4
        laterals[3] = self.mrf_p5(laterals[3])  # 增强P5

        # 4. 标准FPN自顶向下路径（只处理P3-P5）
        used_backbone_levels = len(laterals)

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

        # 4. 应用FPN卷积层并输出P3-P5
        outs = []
        # 输出P3, P4, P5（跳过P2）
        for i in range(1, 4):  # i=1,2,3 对应 P3, P4, P5
            if i < len(laterals):
                out_feat = self.fpn_convs[i - 1](laterals[i])
                outs.append(out_feat)

        # 确保输出3层
        while len(outs) < 3:
            outs.append(outs[-1] if outs else laterals[-1])

        return tuple(outs[:3])  # 返回P3, P4, P5