import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List

class MultiBranchAFE(nn.Module):
    """
    三分支多尺度方向增强模块（无旋转卷积）：
      - 分支1：恒等映射
      - 分支2：1x7 深度可分离卷积（水平）
      - 分支3：7x1 深度可分离卷积（垂直）
    通道分配：恒等占 1/4，两个条形卷积各占 3/8。
    """
    def __init__(self, in_channels, kernel_size=7, stride=1, padding=None):
        super().__init__()
        assert in_channels % 8 == 0, "in_channels must be divisible by 8 for exact 0.25/0.375/0.375 split"
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

        # 计算各分支通道数
        c_id = in_channels // 4                     # 0.25
        c_bar = (in_channels * 3) // 8              # 0.375，注意整数除法，确保整除
        # 保证总和等于 in_channels
        self.c_id = c_id
        self.c_bar = c_bar
        # 实际分配时，恒等分支取 c_id，两个条形分支各取 c_bar，最后可能因为舍入差一两个通道，补到恒等分支
        total = c_id + 2 * c_bar
        if total < in_channels:
            self.c_id += (in_channels - total)
        elif total > in_channels:
            self.c_bar -= (total - in_channels + 1) // 2  # 简单调整

        # 分支2: 1x7 深度可分离卷积
        self.dw_horiz = nn.Conv2d(self.c_bar, self.c_bar,
                                  kernel_size=(1, kernel_size), stride=stride,
                                  padding=(0, padding), groups=self.c_bar, bias=False)
        # 分支3: 7x1 深度可分离卷积
        self.dw_vert = nn.Conv2d(self.c_bar, self.c_bar,
                                 kernel_size=(kernel_size, 1), stride=stride,
                                 padding=(padding, 0), groups=self.c_bar, bias=False)
        # 可选：为条形分支加 BN 和激活（为了公平，也可不加，保持与恒等一致）
        self.bn_horiz = nn.BatchNorm2d(self.c_bar)
        self.bn_vert = nn.BatchNorm2d(self.c_bar)
        self.act = nn.SiLU()

        # 融合卷积（1x1）
        self.fusion = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # 按通道分割
        c_id = self.c_id
        c_bar = self.c_bar
        x_id = x[:, :c_id, :, :]                 # 恒等分支
        x_horiz = x[:, c_id:c_id+c_bar, :, :]    # 水平条形分支
        x_vert = x[:, c_id+c_bar:c_id+2*c_bar, :, :]  # 垂直条形分支

        out_id = x_id
        out_horiz = self.dw_horiz(x_horiz)
        out_vert = self.dw_vert(x_vert)

        out_horiz = self.bn_horiz(out_horiz)
        out_vert = self.bn_vert(out_vert)
        out_horiz = self.act(out_horiz)
        out_vert = self.act(out_vert)

        # 拼接
        out = torch.cat([out_id, out_horiz, out_vert], dim=1)
        # 融合
        out = self.fusion(out)
        out = self.bn_fusion(out)
        out = self.act(out)
        return out


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """
    增强版 FPN，当 fusion_mode 为 'afe' 时使用 MultiBranchAFE（无旋转卷积）。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes: List[str],
                 afe_kernel_size=7,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        self.fusion_modes = fusion_modes
        self.dynamic_convs = nn.ModuleList()
        for mode in fusion_modes:
            if mode == 'afe':
                assert out_channels % 8 == 0, "out_channels must be divisible by 8 for AFE"
                self.dynamic_convs.append(
                    MultiBranchAFE(
                        in_channels=out_channels,
                        kernel_size=afe_kernel_size
                    )
                )
            else:
                self.dynamic_convs.append(None)

    @auto_fp16()
    def forward(self, inputs):
        # 同原始 FPN 的 lateral 构建
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        # 自顶向下融合
        for i in range(used_backbone_levels - 1, 0, -1):
            fusion_idx = used_backbone_levels - 1 - i
            mode = self.fusion_modes[fusion_idx]

            if mode == 'add':
                if 'scale_factor' in self.upsample_cfg:
                    upsampled = F.interpolate(laterals[i], **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
                laterals[i - 1] = laterals[i - 1] + upsampled

            elif mode == 'afe':
                enhanced_low = self.dynamic_convs[fusion_idx](laterals[i - 1])
                up_high = F.interpolate(laterals[i], size=enhanced_low.shape[-2:], **self.upsample_cfg)
                laterals[i - 1] = enhanced_low + up_high

            else:
                raise ValueError(f"Unknown fusion mode: {mode}")

        # 输出层
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # 生成额外层（同原 FPN）
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