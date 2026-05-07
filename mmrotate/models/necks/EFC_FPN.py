import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import NECKS

class GFF(BaseModule):
    """分组特征聚焦单元"""
    def __init__(self, in_channels, out_channels, group_num=16, eps=1e-5):
        super().__init__()
        self.group_num = group_num
        self.eps = eps

        # 对齐低分辨率特征：先上采样到高分辨率尺寸，再1x1调整通道
        self.align_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        # 空间注意力权重生成
        self.spatial_att = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # 分组交互卷积 (每组一个1x1)
        self.group_interact = nn.ModuleList([
            nn.Conv2d(out_channels // group_num, out_channels // group_num, kernel_size=1)
            for _ in range(group_num)
        ])
        # 可学习的仿射参数
        self.gamma = nn.Parameter(torch.ones(out_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, low_feat, high_feat):
        # low_feat: 深层特征, high_feat: 浅层特征（空间尺寸更大）
        # 动态上采样 low_feat 到 high_feat 的尺寸
        target_h, target_w = high_feat.shape[2], high_feat.shape[3]
        low_up = F.interpolate(low_feat, size=(target_h, target_w), mode='nearest')
        low_aligned = self.align_conv(low_up)           # 调整通道

        coarse = low_aligned + high_feat                # 粗融合
        spatial_weight = self.spatial_att(coarse)       # 空间权重
        focused = coarse * spatial_weight               # 空间聚焦

        N, C, H, W = focused.shape
        g_c = C // self.group_num
        groups = focused.chunk(self.group_num, dim=1)   # 分组

        outs = []
        for i, g in enumerate(groups):
            interacted = self.group_interact[i](g)
            # 空间Softmax注意力（基于分组内通道均值）
            att = F.softmax(interacted.mean(dim=1, keepdim=True).view(N,1,-1), dim=-1)
            att = att.view(N,1,H,W)
            outs.append(g * att)
        fused = torch.cat(outs, dim=1)

        # 使用 coarse 的均值和标准差进行归一化
        flat = coarse.view(N, C, -1)
        mean = flat.mean(dim=-1, keepdim=True).view(N,C,1,1)
        std = flat.std(dim=-1, keepdim=True).view(N,C,1,1)
        normalized = (fused - mean) / (std + self.eps)
        out = normalized * self.gamma + self.beta
        return out


class MFR(BaseModule):
    """多级特征重建模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 阈值预测器
        self.threshold_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # 弱特征轻量变换 (深度可分离 + 通道注意力)
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gate_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )
        # 强特征变换 (1x1)
        self.strong_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        coarse = x1 + x2
        T = self.threshold_conv(coarse)              # 阈值
        w1, w2 = torch.sigmoid(x1), torch.sigmoid(x2)  # 重要性权重
        strong = (w1 >= T).float() * coarse + (w2 >= T).float() * coarse
        weak   = (w1 < T).float() * coarse + (w2 < T).float() * coarse

        # 弱特征变换
        weak_t = self.dwconv(weak)
        weak_t = self.pwconv(weak_t)
        gate = self.gate_gen(weak)                   # 通道注意力
        weak_t = weak_t * gate

        # 强特征变换
        strong_t = self.strong_conv(strong)

        out = weak_t + strong_t
        out = self.fusion_conv(out)
        return out


@NECKS.register_module()
class EFC_FPN(BaseModule):
    """EFC特征金字塔（替换FPN）"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 group_num=16):
        super().__init__()
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level if end_level != -1 else len(in_channels)-1
        self.num_ins = len(in_channels)

        # 横向卷积：将backbone各层通道对齐到 out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, 1)
            for i in range(self.start_level, self.end_level+1)
        ])

        # GFF 模块（用于自上而下的相邻层融合）
        self.gff_modules = nn.ModuleList([
            GFF(out_channels, out_channels, group_num)
            for _ in range(self.end_level - self.start_level)
        ])
        # MFR 模块（可选，与GFF并行）
        self.mfr_modules = nn.ModuleList([
            MFR(out_channels, out_channels)
            for _ in range(self.end_level - self.start_level)
        ])

        # 额外输出层（P6, P7）
        self.extra_convs = nn.ModuleList()
        for _ in range(num_outs - (self.end_level - self.start_level + 1)):
            self.extra_convs.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
            )

    @auto_fp16()
    def forward(self, inputs):
        # inputs: (C2, C3, C4, C5) 高分辨率 -> 低分辨率
        laterals = []
        for i, conv in enumerate(self.lateral_convs):
            laterals.append(conv(inputs[self.start_level + i]))

        # 自上而下融合
        out_list = []
        prev = laterals[-1]          # 最顶层（最小分辨率）
        out_list.append(prev)

        # 从顶层向下遍历
        idx = len(laterals) - 2
        gff_idx = len(self.gff_modules) - 1
        while idx >= 0:
            cur = laterals[idx]
            # 上采样 prev 到 cur 的空间尺寸
            prev_up = F.interpolate(prev, size=cur.shape[2:], mode='nearest')
            gff_out = self.gff_modules[gff_idx](prev_up, cur)   # 注意顺序：low_feat, high_feat
            mfr_out = self.mfr_modules[gff_idx](gff_out, cur)
            fused = gff_out + mfr_out
            out_list.insert(0, fused)
            prev = fused
            idx -= 1
            gff_idx -= 1

        # 生成额外输出层（P6, P7）
        extra_outs = []
        for conv in self.extra_convs:
            prev = conv(prev)
            extra_outs.append(prev)

        return tuple(out_list + extra_outs)