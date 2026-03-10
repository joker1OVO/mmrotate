import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import NECKS
from mmdet.models.necks.fpn import FPN
from typing import List, Tuple
import math

# ---------- 辅助函数：生成DCT矩阵 ----------
def get_dct_matrix(n: int, device: torch.device) -> torch.Tensor:
    """生成 n x n 的DCT-II矩阵"""
    k = torch.arange(n, device=device).float().view(-1, 1)
    i = torch.arange(n, device=device).float().view(1, -1)
    matrix = torch.cos(math.pi / n * (i + 0.5) * k)
    scale = torch.ones(n, device=device) * math.sqrt(2.0 / n)
    scale[0] = math.sqrt(1.0 / n)
    matrix = matrix * scale.view(-1, 1)
    return matrix

# ---------- 高频滤波器 ----------
class HighPassFilter(BaseModule):
    """基于DCT的高通滤波器，仅对指定层生效"""
    def __init__(self, alpha: float = 0.25, only_on_layers: List[int] = [0, 1]):
        super().__init__()
        self.alpha = alpha
        self.only_on_layers = only_on_layers
        self._dct_cache = {}

    def _get_matrices(self, h: int, w: int, device: torch.device):
        key = (h, w, device)
        if key not in self._dct_cache:
            T_h = get_dct_matrix(h, device)
            T_w = get_dct_matrix(w, device)
            self._dct_cache[key] = (T_h, T_w)
        return self._dct_cache[key]

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if layer_idx not in self.only_on_layers:
            return x
        B, C, H, W = x.shape
        device = x.device
        T_h, T_w = self._get_matrices(H, W, device)

        x_reshaped = x.view(B * C, H, W)
        dct_h = torch.matmul(T_h, x_reshaped)
        dct_hw = torch.matmul(T_w, dct_h.permute(0, 2, 1))
        dct_coeff = dct_hw.permute(0, 2, 1)

        h_cut = int(self.alpha * H)
        w_cut = int(self.alpha * W)
        mask = torch.ones((H, W), device=device)
        mask[:h_cut, :w_cut] = 0
        mask = mask.view(1, 1, H, W).expand(B, C, -1, -1).reshape(B * C, H, W)
        dct_coeff = dct_coeff * mask

        idct_w = torch.matmul(T_w.t(), dct_coeff.permute(0, 2, 1))
        idct_wh = torch.matmul(T_h.t(), idct_w.permute(0, 2, 1))
        out = idct_wh.view(B, C, H, W)
        return out

# ---------- 通道路径 ----------
class ChannelPath(BaseModule):
    def __init__(self, channels: int, k: int = 16, conv_cfg=None, norm_cfg=None, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.k = k
        self.pool_avg = nn.AdaptiveAvgPool2d((k, k))
        self.pool_max = nn.AdaptiveMaxPool2d((k, k))
        self.conv1_avg = ConvModule(channels, channels, 1,
                                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1_max = ConvModule(channels, channels, 1,
                                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.fusion_conv = ConvModule(2 * channels, channels, 1,
                                      conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        B, C, H, W = f.shape
        avg_pool = self.pool_avg(f)
        max_pool = self.pool_max(f)

        avg_feat = self.conv1_avg(avg_pool)
        max_feat = self.conv1_max(max_pool)

        avg_vec = avg_feat.sum(dim=(2, 3))
        max_vec = max_feat.sum(dim=(2, 3))

        cat_vec = torch.cat([avg_vec, max_vec], dim=1).view(B, 2*C, 1, 1)
        weight = self.fusion_conv(cat_vec)
        return weight

# ---------- 空间路径 ----------
class SpatialPath(BaseModule):
    def __init__(self, channels: int, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv = ConvModule(channels, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.conv(f)

# ---------- HFP模块 ----------
class HFPModule(BaseModule):
    def __init__(self,
                 channels: int,
                 alpha: float = 0.25,
                 k: int = 16,
                 only_filter_layers: List[int] = [0, 1],
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.channels = channels
        self.hpf = HighPassFilter(alpha=alpha, only_on_layers=only_filter_layers)
        self.cp = ChannelPath(channels, k=k, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.sp = SpatialPath(channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        self.conv_out = ConvModule(channels, channels, 3, padding=1,
                                   conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        f = self.hpf(x, layer_idx)
        w_c = self.cp(f)
        w_s = self.sp(f)
        out = x * w_c + x * w_s
        out = self.conv_out(out)
        return out

# ---------- SDP模块 ----------
class SpatialDependencyPerception(BaseModule):
    def __init__(self, channels: int, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv_q = ConvModule(channels, channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_k = ConvModule(channels, channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_v = ConvModule(channels, channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_out = ConvModule(channels, channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, c: torch.Tensor, p_up: torch.Tensor, block_size: Tuple[int, int]) -> torch.Tensor:
        B, C, H, W = c.shape
        block_h, block_w = block_size
        assert H % block_h == 0 and W % block_w == 0, \
            f"输入尺寸 ({H},{W}) 必须能被块大小 ({block_h},{block_w}) 整除"
        n_h = H // block_h
        n_w = W // block_w
        n_blocks = n_h * n_w
        L = block_h * block_w

        Q = self.conv_q(c)
        K = self.conv_k(p_up)
        V = self.conv_v(p_up)

        Q = Q.view(B, C, n_h, block_h, n_w, block_w).permute(0, 2, 4, 3, 5, 1).contiguous()
        K = K.view(B, C, n_h, block_h, n_w, block_w).permute(0, 2, 4, 3, 5, 1).contiguous()
        V = V.view(B, C, n_h, block_h, n_w, block_w).permute(0, 2, 4, 3, 5, 1).contiguous()
        Q = Q.view(B, n_blocks, L, C)
        K = K.view(B, n_blocks, L, C)
        V = V.view(B, n_blocks, L, C)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)

        out = out.view(B, n_h, n_w, block_h, block_w, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(B, C, H, W)
        out = self.conv_out(out)
        return out

# ---------- HS-FPN主类 ----------
@NECKS.register_module()
class HSFPN(FPN):
    """
    HS-FPN，支持5层输出（P2~P6）。
    继承自MMDetection的FPN，复用其lateral_convs, fpn_convs。
    在生成额外输出时，若`extra_convs`属性不存在，则使用最大池化。
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 alpha: float = 0.25,
                 k: int = 16,
                 only_filter_layers: List[int] = [0, 1],
                 with_sdp: bool = True,
                 sdp_layers: List[int] = [0, 1, 2],
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.num_ins = len(in_channels)
        self.with_sdp = with_sdp
        self.sdp_layer_indices = set(sdp_layers)

        # HFP模块
        self.hfp_modules = nn.ModuleList()
        for i in range(self.num_ins):
            self.hfp_modules.append(
                HFPModule(
                    out_channels,
                    alpha=alpha,
                    k=k,
                    only_filter_layers=only_filter_layers,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # SDP模块
        if self.with_sdp:
            self.sdp_modules = nn.ModuleList()
            for i in range(self.num_ins - 1):
                if i in self.sdp_layer_indices:
                    self.sdp_modules.append(
                        SpatialDependencyPerception(
                            out_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
                else:
                    self.sdp_modules.append(nn.Identity())
        else:
            self.sdp_modules = nn.ModuleList([nn.Identity() for _ in range(self.num_ins - 1)])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        super().init_weights()

    @auto_fp16()
    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        assert len(inputs) == self.num_ins

        # 1. 通过lateral_convs降维
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # 2. 应用HFP
        hfp_outs = []
        for i, lateral in enumerate(laterals):
            hfp_out = self.hfp_modules[i](lateral, layer_idx=i)
            hfp_outs.append(hfp_out)

        # 3. Top-down + SDP
        inner_outs = [None] * self.num_ins
        top_size = hfp_outs[-1].shape[-2:]   # 最高层尺寸作为块大小基准
        for i in range(self.num_ins - 1, -1, -1):
            if i == self.num_ins - 1:
                inner_outs[i] = hfp_outs[i]
            else:
                higher = F.interpolate(inner_outs[i+1], size=hfp_outs[i].shape[-2:], mode='nearest')
                if self.with_sdp and i < len(self.sdp_modules) and i in self.sdp_layer_indices:
                    sdp_out = self.sdp_modules[i](hfp_outs[i], higher, top_size)
                    fused = hfp_outs[i] + sdp_out
                else:
                    fused = hfp_outs[i] + higher
                inner_outs[i] = fused

        # 4. 应用fpn_convs得到基础输出
        outs = []
        for i in range(self.num_ins):
            outs.append(self.fpn_convs[i](inner_outs[i]))

        # 5. 生成额外输出（如P6）
        if self.num_outs > len(outs):
            # 如果父类没有创建 extra_convs 属性，直接使用最大池化
            if not hasattr(self, 'extra_convs') or self.extra_convs is None:
                for i in range(self.num_outs - len(outs)):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                # 确定 extra_source
                if hasattr(self, 'add_extra_convs'):
                    if self.add_extra_convs == 'on_input':
                        extra_source = inputs[-1]
                    elif self.add_extra_convs == 'on_output':
                        extra_source = outs[-1]
                    elif self.add_extra_convs == 'on_lateral':
                        extra_source = laterals[-1]
                    else:
                        extra_source = outs[-1]
                else:
                    extra_source = outs[-1]  # fallback

                # 遍历 extra_convs
                for i, extra_conv in enumerate(self.extra_convs):
                    if i == 0:
                        if extra_conv is not None:
                            outs.append(extra_conv(extra_source))
                        else:
                            outs.append(F.max_pool2d(extra_source, 1, stride=2))
                    else:
                        if extra_conv is not None:
                            outs.append(extra_conv(outs[-1]))
                        else:
                            outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        return tuple(outs)