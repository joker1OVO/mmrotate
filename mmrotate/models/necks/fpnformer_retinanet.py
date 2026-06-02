import math
from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import ROTATED_NECKS


# ---------------------------------------------------------------------------
# positional encoding
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding for 2D feature maps."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)),
                               device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32,
                             device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ---------------------------------------------------------------------------
# window-based self-attention wrapper
# ---------------------------------------------------------------------------

class Swin(nn.Module):
    """Thin wrapper around torchvision's SwinTransformerBlock."""

    def __init__(self, window_size, shift_size=None):
        super().__init__()
        if shift_size is None:
            shift_size = [0, 0]
        from torchvision.models.swin_transformer import SwinTransformerBlock
        self.encoder = SwinTransformerBlock(
            dim=256, num_heads=8,
            window_size=window_size,
            shift_size=shift_size)

    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.encoder(x)
        # [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x


# ---------------------------------------------------------------------------
# Transformer layers for cross-scale decoding
# ---------------------------------------------------------------------------

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask=None,
                     memory_key_padding_mask=None,
                     pos=None,
                     query_pos=None):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None,
                    query_pos=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=1024, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


# ---------------------------------------------------------------------------
# Cross-scale decoder block
# ---------------------------------------------------------------------------

class Fpndecoder(nn.Module):
    """One cross-scale decoder: q attends to kv via window-SA + cross-attn + FFN.

    Position embeddings are computed dynamically from the input tensors,
    so the module adapts to any feature-map size.
    """

    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.selfatt = Swin([7, 7])
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=d_model // 2)
        self.cross = CrossAttentionLayer(d_model=d_model, nhead=nhead)
        self.ffn = FFNLayer(d_model=d_model)

    def forward(self, q, kv):
        """
        Args:
            q:  [B, C, H, W]  query feature
            kv: [B, C, Hv, Wv] key/value feature (may differ in HW)
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = q.shape

        # dynamic position embeddings
        q_pos = self.pos_embed(q).flatten(2).transpose(1, 2)   # [B, HW, C]
        kv_pos = self.pos_embed(kv).flatten(2).transpose(1, 2)  # [B, HvWv, C]

        q = self.selfatt(q)  # [B, C, H, W]

        q = q.flatten(2).transpose(1, 2)    # [B, HW, C]
        kv = kv.flatten(2).transpose(1, 2)  # [B, HvWv, C]

        q = self.cross(q, kv, query_pos=q_pos, pos=kv_pos)
        q = self.ffn(q)  # [B, HW, C]

        q = q.transpose(1, 2).reshape(B, C, H, W)
        return q


# ---------------------------------------------------------------------------
# Main neck
# ---------------------------------------------------------------------------

@ROTATED_NECKS.register_module()
class FPNdecoderformer_swin_double(BaseModule):
    """FPN with two rounds of cross-scale transformer decoding.

    Builds a standard FPN, then refines C3/C4/C5 through two iterative
    rounds of cross-attention: each level attends to another level
    (e.g. C3 queries C5, C5 queries C3).

    Args:
        in_channels (list[int]): Input channels per backbone level.
        out_channels (int): Output channels (all levels).
        num_outs (int): Number of output levels (>= 3).
        start_level (int): Index of the first backbone level used. Default: 0.
        end_level (int): Index of the last backbone level used. Default: -1.
        add_extra_convs (bool | str): Extra P6+ levels. Default: False.
        relu_before_extra_convs (bool): ReLU before extra convs. Default: False.
        no_norm_on_lateral (bool): Skip norm on lateral convs. Default: False.
        conv_cfg (dict): Config for ConvModule.
        norm_cfg (dict): Config for norm layers.
        act_cfg (dict): Config for activation layers.
        upsample_cfg (dict): Config for F.interpolate. Default: dict(mode='nearest').
        init_cfg (dict): Weight init config.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs: Union[bool, str] = False,
                 relu_before_extra_convs: bool = False,
                 no_norm_on_lateral: bool = False,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 upsample_cfg: dict = None,
                 init_cfg: Optional[dict] = None):
        if init_cfg is None:
            init_cfg = dict(type='Xavier', layer='Conv2d', distribution='uniform')
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = (upsample_cfg.copy() if upsample_cfg is not None
                             else dict(mode='nearest'))

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_extra = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_extra = out_channels
                extra_fpn_conv = ConvModule(
                    in_extra,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        # cross-scale decoders: two rounds of C3↔C4↔C5 refinement
        self.decoder_c5_c4 = Fpndecoder(d_model=out_channels)
        self.decoder_c5_c3 = Fpndecoder(d_model=out_channels)
        self.decoder_c4_c5 = Fpndecoder(d_model=out_channels)
        self.decoder_c3_c5 = Fpndecoder(d_model=out_channels)
        self.decoder_c5_c4_2 = Fpndecoder(d_model=out_channels)
        self.decoder_c5_c3_2 = Fpndecoder(d_model=out_channels)
        self.decoder_c4_c5_2 = Fpndecoder(d_model=out_channels)
        self.decoder_c3_c5_2 = Fpndecoder(d_model=out_channels)

    @auto_fp16()
    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        assert len(inputs) == len(self.in_channels)

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

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

        # ---- cross-scale decoding (two rounds) ----
        c3, c4, c5 = outs[0], outs[1], outs[2]

        # round 1
        c3 = self.decoder_c3_c5(c3, c5)
        c4 = self.decoder_c4_c5(c4, c5)
        c5 = self.decoder_c5_c4(c5, c4)
        c5 = self.decoder_c5_c3(c5, c3)
        # round 2
        c3 = self.decoder_c3_c5_2(c3, c5)
        c4 = self.decoder_c4_c5_2(c4, c5)
        c5 = self.decoder_c5_c4_2(c5, c4)
        c5 = self.decoder_c5_c3_2(c5, c3)

        outs[0], outs[1], outs[2] = c3.contiguous(), c4.contiguous(), c5.contiguous()

        return tuple(outs)
