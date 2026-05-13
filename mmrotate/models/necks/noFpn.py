import torch.nn as nn
from mmcv.cnn import ConvModule
from ..builder import ROTATED_NECKS

@ROTATED_NECKS.register_module()
class noFpn(nn.Module):
    """仅统一通道数并生成第5层，不进行任何融合的多尺度neck。

    Args:
        in_channels (list[int]): 骨干各层输出通道数，如 [256,512,1024,2048]
        out_channels (int): 统一后的通道数，通常为256
        conv_cfg (dict): 卷积配置
        norm_cfg (dict): 归一化配置
        act_cfg (dict): 激活函数配置
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)   # 应该是4

        # 为每个输入层创建一个1x1卷积，将通道数统一到out_channels
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        # 用于生成第5层（P6）的卷积，步长为2的3x3卷积
        self.extra_convs = ConvModule(
            out_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

    def forward(self, inputs):
        """前向传播，返回统一通道后的多尺度特征列表，顺序从高分辨率到低分辨率，共5层。"""
        assert len(inputs) == self.num_ins   # inputs 长度应为4
        outs = []
        for i, x in enumerate(inputs):
            out = self.lateral_convs[i](x)
            outs.append(out)

        # 生成第5层（P6）：对最后一层（stride最大）进行步长为2的卷积下采样
        p6 = self.extra_convs(outs[-1])
        outs.append(p6)

        return tuple(outs)