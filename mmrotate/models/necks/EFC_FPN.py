import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import NECKS


class AttentionMask(nn.Module):
    """生成空间注意力掩码（模仿 GFLDYHead 中的 gfl_cls_mask / gfl_reg_mask）"""
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class ChannelGate(nn.Module):
    """通道注意力门控（模仿 GFLDYHead 中的 gate_genator）"""
    def __init__(self, channels, reduction=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weight = self.fc(self.gap(x))
        return x * weight


class DynamicConvBlock(nn.Module):
    """轻量化动态卷积块：包含空间掩码、pointwise 特征和普通卷积（代替 DyConv2D）"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.mask_att = AttentionMask(in_channels, kernel_size)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1)  # pointwise 特征分支

    def forward(self, x, mask=None, pw=None):
        # x: 输入特征; mask: 空间掩码; pw: pointwise 特征（可选）
        if mask is None:
            mask = self.mask_att(x)
        # 普通卷积
        out = self.conv(x)
        # 应用空间掩码（逐像素加权）
        out = out * mask
        # 融合 pointwise 特征
        if pw is not None:
            out = out + self.pw_conv(pw)
        return out


class EFCFusion(BaseModule):
    """EFC 融合模块（集成 GFF + MFR 思想，并加入动态增强）"""
    def __init__(self, in_channels, out_channels, group_num=16, eps=1e-5):
        super().__init__()
        self.group_num = group_num
        self.eps = eps

        # 对齐卷积（1x1）
        self.align_conv = nn.Conv2d(in_channels, out_channels, 1)

        # 空间掩码生成器
        self.spatial_mask = AttentionMask(out_channels)

        # 分组交互卷积（4 组，每组独立 1x1）
        self.group_interacts = nn.ModuleList([
            nn.Conv2d(out_channels // 4, out_channels // 4, 1)
            for _ in range(4)
        ])

        # 可学习仿射参数
        self.gamma = nn.Parameter(torch.randn(out_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(out_channels, 1, 1))

        # 强弱特征分离与轻量变换
        self.gate_gen = ChannelGate(out_channels)
        self.dwconv = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels)
        self.pwconv = nn.Conv2d(out_channels, out_channels, 1)
        self.strong_conv = nn.Conv2d(out_channels, out_channels, 1)

        # 动态卷积块（最后增强）
        self.dynamic_block = DynamicConvBlock(out_channels, out_channels)

    def forward(self, low_feat, high_feat):
        # low_feat: 深层特征（小分辨率）, high_feat: 浅层特征（大分辨率）
        # 上采样低分辨率特征
        target_h, target_w = high_feat.shape[2], high_feat.shape[3]
        low_up = F.interpolate(low_feat, size=(target_h, target_w), mode='nearest')
        low_aligned = self.align_conv(low_up)   # 通道对齐

        # 粗融合
        coarse = low_aligned + high_feat

        # 空间注意力掩码
        mask = self.spatial_mask(coarse)          # (N,1,H,W)
        masked = coarse * mask

        # 分组交互（分为4组，通道方向）
        groups = masked.chunk(4, dim=1)          # list of (N, C//4, H, W)
        refined = []
        for i, g in enumerate(groups):
            interacted = self.group_interacts[i](g)
            # 空间-通道联合注意力（简化版 softmax 归一化）
            N, Cg, H, W = interacted.shape
            flat = interacted.view(N, 1, -1)
            mean_flat = flat.mean(dim=2, keepdim=True) + self.eps
            norm_flat = flat / mean_flat
            att = F.softmax(norm_flat, dim=-1).view(N, Cg, H, W)
            refined.append(g * att)
        fused = torch.cat(refined, dim=1)        # (N, C, H, W)

        # 分组归一化（参照 coarse 的统计量）
        N, C, H, W = fused.shape
        # 将 coarse 和 fused 都分成 group_num 组
        coarse_reshaped = coarse.view(N, self.group_num, -1)
        fused_reshaped = fused.view(N, self.group_num, -1)
        mean_c = coarse_reshaped.mean(dim=2, keepdim=True)
        std_c = coarse_reshaped.std(dim=2, keepdim=True)
        normed = (fused_reshaped - mean_c) / (std_c + self.eps)
        normed = normed.view(N, C, H, W)
        normed = normed * self.gamma + self.beta   # 仿射

        # -------------------- MFR 部分 --------------------
        # 计算强弱分离阈值（全局平均池化）
        global_avg = torch.mean(coarse, dim=[2,3], keepdim=True)   # (N,C,1,1)
        threshold = torch.sigmoid(global_avg)                        # (N,C,1,1)

        # 计算两个输入特征的权重（文中用 BN+sigmoid，这里简化）
        w_high = torch.sigmoid(high_feat)
        w_low  = torch.sigmoid(low_aligned)

        strong_mask_high = (w_high >= threshold).float()
        weak_mask_high   = (w_high < threshold).float()
        strong_mask_low  = (w_low  >= threshold).float()
        weak_mask_low    = (w_low  < threshold).float()

        strong_feat = strong_mask_high * coarse + strong_mask_low * coarse
        weak_feat   = weak_mask_high   * coarse + weak_mask_low   * coarse

        # 弱特征轻量变换
        weak_out = self.dwconv(weak_feat)
        weak_out = self.pwconv(weak_out)
        weak_out = self.gate_gen(weak_out)          # 通道注意力

        # 强特征简单变换
        strong_out = self.strong_conv(strong_feat)

        # 动态卷积增强（最终输出）
        out = weak_out + strong_out + normed
        out = self.dynamic_block(out, mask=mask)    # 最后一次动态增强

        return out


@NECKS.register_module()
class EFC_FPN(BaseModule):
    """完全体 EFC 特征金字塔（基于 GFLDYHead 中的动态增强思想）"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 start_level=1,
                 end_level=-1,
                 group_num=16):
        super().__init__()
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level if end_level != -1 else len(in_channels) - 1
        self.num_ins = len(in_channels)

        # 横向卷积（将 backbone 输出通道统一到 out_channels）
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, 1)
            for i in range(self.start_level, self.end_level + 1)
        ])

        # EFC 融合模块（数量 = 横向层数 - 1）
        self.efc_blocks = nn.ModuleList([
            EFCFusion(out_channels, out_channels, group_num)
            for _ in range(len(self.lateral_convs) - 1)
        ])

        # 额外输出层（P6, P7）使用 Maxpool + 1x1
        self.extra_convs = nn.ModuleList()
        for _ in range(num_outs - len(self.lateral_convs)):
            self.extra_convs.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(out_channels, out_channels, 1)
                )
            )

    @auto_fp16()
    def forward(self, inputs):
        # inputs: (C2, C3, C4, C5) 高分辨率到低分辨率
        laterals = []
        for i, conv in enumerate(self.lateral_convs):
            laterals.append(conv(inputs[self.start_level + i]))

        # 自顶向下融合
        out_list = []
        prev = laterals[-1]          # 最顶层（最小分辨率）
        out_list.append(prev)

        for idx in range(len(laterals)-2, -1, -1):
            cur = laterals[idx]
            # 上采样 prev 到 cur 的尺寸
            prev_up = F.interpolate(prev, size=cur.shape[2:], mode='nearest')
            fused = self.efc_blocks[idx](prev_up, cur)   # 注意顺序：low_feat, high_feat
            out_list.insert(0, fused)
            prev = fused

        # 生成额外输出（P6, P7）
        extra_outs = []
        for conv in self.extra_convs:
            prev = conv(prev)
            extra_outs.append(prev)

        return tuple(out_list + extra_outs)