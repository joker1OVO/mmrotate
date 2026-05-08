# mmrotate/models/necks/re_fpn_mrf45.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import ROTATED_NECKS
from ..SAFF_DETR import MRF, RepC3   # 假设 MRF 和 RepC3 已定义


@ROTATED_NECKS.register_module()
class mrf(BaseModule):
    """
    对 P2-P5 四层分别进行 MRF 增强，然后进行 FPN 自顶向下融合，输出 P2-P5。

    Args:
        in_channels (list[int]): 输入通道数，[C2, C3, C4, C5]
        out_channels (int): 输出通道数（固定为 256）
        num_outs (int): 输出层数，固定为 4（P2, P3, P4, P5）
        start_level (int): 起始 backbone 层索引，默认 0
        end_level (int): 结束 backbone 层索引，默认 -1
        add_extra_convs (bool): 是否添加额外卷积，本实现中忽略
        relu_before_extra_convs (bool): 忽略
        no_norm_on_lateral (bool): 侧向连接是否使用 Norm
        conv_cfg, norm_cfg, act_cfg: 卷积配置
        upsample_cfg: 上采样配置
        init_cfg: 初始化配置
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=4,
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
        super(mrf, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        assert len(in_channels) == 4, "需要4个输入特征层 [C2, C3, C4, C5]"
        assert num_outs == 5, "输出必须为4层 [P2, P3, P4, P5]"

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        # ==================== 1. 侧向卷积（降维） ====================
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

        # ==================== 2. 为 P2,P3,P4,P5 分别创建 MRF 增强模块 ====================
        # MRF 输入输出通道均为 out_channels
        self.mrf_p2 = MRF(dim=out_channels)
        self.mrf_p3 = MRF(dim=out_channels)
        self.mrf_p4 = MRF(dim=out_channels)
        self.mrf_p5 = MRF(dim=out_channels)

        # ==================== 3. FPN 输出卷积层（对融合后的 P2-P5 分别应用 3x3 conv） ====================
        self.fpn_convs = nn.ModuleList()
        for _ in range(self.num_outs):   # 4 个卷积层
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
        """
        Processing flow:
        1. 侧向卷积降维得到 laterals (P2_raw, P3_raw, P4_raw, P5_raw)
        2. 对每个 laterals 独立应用 MRF 增强 (P2_enh, P3_enh, P4_enh, P5_enh)
        3. FPN 自顶向下融合：
            P5_fused = P5_enh
            P4_fused = P4_enh + upsample(P5_fused)
            P3_fused = P3_enh + upsample(P4_fused)
            P2_fused = P2_enh + upsample(P3_fused)
        4. 对融合后的每一层应用 3x3 卷积，输出 P2, P3, P4, P5
        """
        assert len(inputs) == len(self.in_channels)

        # Step 1: 降维
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # laterals[0] = P2_raw, laterals[1] = P3_raw, laterals[2] = P4_raw, laterals[3] = P5_raw

        # Step 2: 各层独立 MRF 增强
        p2_enh = self.mrf_p2(laterals[0])
        p3_enh = self.mrf_p3(laterals[1])
        p4_enh = self.mrf_p4(laterals[2])
        p5_enh = self.mrf_p5(laterals[3])

        # Step 3: FPN 自顶向下融合（无下采样，只有上采样加法）
        # 从最高层 P5 开始
        p5_fused = p5_enh
        # P4 ← P4_enh + upsample(P5_fused)
        p4_fused = p4_enh + F.interpolate(p5_fused, size=p4_enh.shape[2:], **self.upsample_cfg)
        # P3 ← P3_enh + upsample(P4_fused)
        p3_fused = p3_enh + F.interpolate(p4_fused, size=p3_enh.shape[2:], **self.upsample_cfg)
        # P2 ← P2_enh + upsample(P3_fused)
        p2_fused = p2_enh + F.interpolate(p3_fused, size=p2_enh.shape[2:], **self.upsample_cfg)

        # Step 4: 应用输出卷积
        outs = []
        fused_list = [p2_fused, p3_fused, p4_fused, p5_fused]
        for i, feat in enumerate(fused_list):
            out = self.fpn_convs[i](feat)
            outs.append(out)

        return tuple(outs)   # (P2, P3, P4, P5)