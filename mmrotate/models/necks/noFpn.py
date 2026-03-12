_base_ = [...]
# 在文件开头或model前添加自定义neck类
from mmdet.models.builder import NECKS
import torch.nn as nn
@NECKS.register_module()
class noFpn(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        return inputs   # 直接返回骨干的多尺度特征