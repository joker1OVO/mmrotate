# mmrotate/models/necks/saff_neck.py

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16

from mmrotate.models.builder import ROTATED_NECKS

# 导入依赖模块
from ..SAFF_DETR import (AIFI, MRF, RepC3,Conv,
                         CrossLayerChannelWiseFrequencyAttention,
                         CrossLayerSpatialWiseFrequencyAttention)


@ROTATED_NECKS.register_module()
class SAFFNeck(BaseModule):
    """SAFF-DETR Neck for Oriented RCNN - 修正版本

    严格遵循 SAFF-DETR 原始设计：
    - 输入: ResNet的C2-C5 (4层特征)
    - 融合: P2→P3 自底向上融合
    - 输出: P3、P4、P5 三层特征图

    Args:
        in_channels (List[int]): ResNet输入特征通道数 [256, 512, 1024, 2048]
        out_channels (int): 输出特征图统一通道数
        num_outs (int): 输出特征图数量（固定为3）
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 num_outs=3,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,  # 新增：与FPN保持一致
                 relu_before_extra_convs=False,  # 新增：与FPN保持一致
                 no_norm_on_lateral=False,  # 新增：与FPN保持一致
                 conv_cfg=None,
                 norm_cfg=None,  # 改为None，与FPN保持一致
                 act_cfg=None,  # 改为None，与FPN保持一致
                 upsample_cfg=dict(mode='nearest'),  # 新增：与FPN保持一致
                 init_cfg=dict(type='Xavier',layer='Conv2d',distribution='uniform')):  # 新增：与FPN保持一致

        super(SAFFNeck, self).__init__(init_cfg)

        # 参数验证
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()  # 与FPN保持一致
        assert len(in_channels) == 4, f"SAFFNeck需要4个输入特征层级 [C2, C3, C4, C5]，但得到 {len(in_channels)} 个"
        assert num_outs == 3, f"SAFFNeck设计为输出3个特征层级 [P3, P4, P5]，但配置为 {num_outs}"
        # 设置输入层级范围
        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
        self.start_level = start_level
        self.end_level = end_level

        # ==================== 1. 特征降维卷积 ====================
        # 将ResNet的4个输出层统一到256通道
        self.reduce_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            input_channels = self.in_channels[i]
            reduce_conv = nn.Sequential(
                nn.Conv2d(input_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
            self.reduce_convs.append(reduce_conv)

        # ==================== 2. AIFI 全局注意力模块 ====================
        # 对C5(P5)应用全局注意力
        self.aifi = AIFI(
            c1=out_channels,
            cm=out_channels * 8,  # 2048 = 256 * 8
            num_heads=8
        )

        # AIFI后处理卷积
        self.post_aifi_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

        # ==================== 3. 自底向上融合路径 ====================
        # P2→P3融合（对应yaml中的第14-17层）

        # P2下采样到P3分辨率（4× → 8×）
        self.p2_downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

        # 多分支表示融合 (MRF)
        self.mrf_p23 = MRF(dim=out_channels * 2)

        # RepC3 高效特征提取（对应yaml中的3次重复）
        self.rep_c3_p3 = RepC3(
            c1=out_channels * 2,  # P2下采样 + P3
            c2=out_channels,
            n=3,  # 3次重复，对应yaml中的3
            e=0.5
        )

        # ==================== 4. 交叉层频率注意力模块 ====================
        # 通道注意力（对应yaml中的CrossLayerChannelHiLoAttention）
        self.cross_channel_attn = CrossLayerChannelWiseFrequencyAttention(
            in_dim=out_channels,   # 单个整数，不是列表
            layer_num=3,           # 3层特征
            reduction=4,            # 缩减比例
            spatial_size = 1024
        )

        # 空间注意力（对应yaml中的CrossLayerHiLoSpatialAttention）
        self.cross_spatial_attn = CrossLayerSpatialWiseFrequencyAttention(
            in_dim=out_channels,   # 单个整数，不是列表
            layer_num=3,           # 3层特征
            reduction=4,            # 缩减比例
            spatial_size=1024
        )

    @auto_fp16()
    @auto_fp16()
    def forward(self, inputs):
        """前向传播 - 严格遵循SAFF-DETR原始结构

        Args:
            inputs (list[Tensor]): 来自ResNet的4个特征图
                - inputs[0]: C2 (256通道，1/4下采样) - P2
                - inputs[1]: C3 (512通道，1/8下采样) - P3
                - inputs[2]: C4 (1024通道，1/16下采样) - P4
                - inputs[3]: C5 (2048通道，1/32下采样) - P5

        Returns:
            tuple[Tensor]: 增强后的3个特征图 (P3, P4, P5)
        """
        # ==================== 阶段1: 特征降维 ====================
        # 将ResNet的C2-C5降维到统一通道数
        p2_reduced = self.reduce_convs[0](inputs[0])  # C2 → P2 (1/4) 13
        p3_reduced = self.reduce_convs[1](inputs[1])  # C3 → P3 (1/8) 12
        p4_reduced = self.reduce_convs[2](inputs[2])  # C4 → P4 (1/16) 11
        p5_reduced = self.reduce_convs[3](inputs[3])  # C5 → P5 (1/32) 8
        # ==================== 阶段2: 高层特征增强 ====================
        p5_aifi = self.aifi(p5_reduced)  # 9
        p5_enhanced = self.post_aifi_conv(p5_aifi)  # 10
        # ==================== 阶段3: 自底向上融合 ====================
        p2_down = self.p2_downsample(p2_reduced)  # 14
        p23_concat = torch.cat([p2_down, p3_reduced], dim=1)  # 15
        p23_mrf = self.mrf_p23(p23_concat)  # 16
        p3_enhanced = self.rep_c3_p3(p23_mrf)  # 17
        # ==================== 阶段4: 交叉层注意力融合 ====================
        channel_attn_inputs = [p3_enhanced, p4_reduced, p5_enhanced]  # 18
        channel_attn_output = self.cross_channel_attn(channel_attn_inputs)
        if isinstance(channel_attn_output, list) and len(channel_attn_output) == 3:  #19-21
            p3_channel, p4_channel, p5_channel = channel_attn_output
        else:
            # 如果输出不是列表，尝试转换为列表
            p3_channel, p4_channel, p5_channel = self._extract_multi_scale_features(channel_attn_output, 3)

        spatial_attn_inputs = [p3_channel, p4_channel, p5_channel]  #22
        spatial_attn_output = self.cross_spatial_attn(spatial_attn_inputs)
        if isinstance(spatial_attn_output, list) and len(spatial_attn_output) == 3:  #23-25
            p3_spatial, p4_spatial, p5_spatial = spatial_attn_output
        else:
            # 如果输出不是列表，尝试转换为列表
            p3_spatial, p4_spatial, p5_spatial = self._extract_multi_scale_features(spatial_attn_output, 3)
        # ==================== 阶段5: 残差连接和最终输出 ====================
        p3_final = p3_enhanced + p3_spatial  #26
        p4_final = p4_reduced + p4_spatial  #27
        p5_final = p5_enhanced + p5_spatial  #28
        # 返回3层特征图 [P3, P4, P5]
        return (p3_final, p4_final, p5_final)

    def _extract_multi_scale_features(self, attn_output, num_scales=3):
        """从注意力输出中提取多尺度特征

        备用方法：用于处理非列表格式的输出

        Args:
            attn_output (Tensor): 注意力模块的输出张量
            num_scales (int): 需要提取的尺度数量

        Returns:
            tuple[Tensor]: 多个尺度的特征图
        """
        # 如果已经是列表或元组，直接返回
        if isinstance(attn_output, (list, tuple)):
            assert len(attn_output) == num_scales, f"多尺度注意力输出应该包含{num_scales}个特征图"
            return attn_output

        # 如果是张量，检查形状并尝试分割
        if isinstance(attn_output, torch.Tensor):
            if attn_output.dim() == 4:  # [B, C, H, W] 格式
                # 如果是单个特征图，复制为多个相同特征图
                # 注意：这只是一个备用方案
                return (attn_output, attn_output, attn_output)

        # 其他情况，返回相同特征图作为备份
        return (attn_output, attn_output, attn_output)

    def init_weights(self):
        """初始化权重"""
        super(SAFFNeck, self).init_weights()

        # 对卷积层使用Kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
