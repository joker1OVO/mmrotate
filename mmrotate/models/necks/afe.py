import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List


class CoordAttention(nn.Module):
    """Coordinate Attention for spatial localization enhancement.

    As described in the ARFN-MoE paper (Eq. 11-16):
        Pool H/W → Concat → Conv(1×1) → Split → Conv(1×1) per branch
        → Sigmoid → element-wise multiply

    No BN, no bottleneck — strictly follows the paper.

    Args:
        channels (int): Input channels.
    """

    def __init__(self, channels):
        super().__init__()
        # Paper Eq.12-13: single 1×1 conv fuses H+W information
        self.fuse_conv = nn.Conv2d(channels, channels, 1, bias=False)
        # Paper Eq.14-15: separate 1×1 convs for H and W attention
        self.conv_h = nn.Conv2d(channels, channels, 1)
        self.conv_w = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Paper Eq.11: pool along H and W separately
        x_h = x.mean(dim=2, keepdim=True)       # [B, C, 1, W]
        x_w = x.mean(dim=3, keepdim=True)       # [B, C, H, 1]
        x_w = x_w.permute(0, 1, 3, 2)            # [B, C, 1, H]

        # Concat → 1×1 Conv (Eq.11-12)
        cat = torch.cat([x_h, x_w], dim=3)       # [B, C, 1, W+H]
        fused = self.fuse_conv(cat)               # [B, C, 1, W+H]

        # Split H/W (Eq.12-13)
        h_feat = fused[:, :, :, :W]               # [B, C, 1, W]
        w_feat = fused[:, :, :, W:]               # [B, C, 1, H]

        # 1×1 Conv + Sigmoid per branch (Eq.14-15)
        ca_h = self.conv_h(h_feat).sigmoid()      # [B, C, 1, W]
        ca_w = self.conv_w(w_feat).sigmoid()      # [B, C, 1, H]

        # Element-wise multiply (Eq.16)
        return x * ca_h * ca_w.permute(0, 1, 3, 2)


class ARFCBlock(nn.Module):
    """Adaptive Receptive Field Convolution Block.

    Strictly follows the ARFN-MoE paper (TGRS 2026):

    - LCE:  AvgPool → DWConv(1×K) → DWConv(K×1) → Conv(1×1)
    - MFE:  N experts, each = pointwise reduce → DWConv(k×k) → CoordAttention
            → pointwise expand. Kernel sizes and channel counts differ per expert.
    - Grid Router: cosine similarity + Top-k sparse selection
    - Balance Loss: CV²
    - Output: LCE(x) + Σ router_i·MFE_i(x) + x  (residual, no BN/activation)

    Args:
        channels (int): Input/output channels C.
        num_experts (int): Number of MFE experts. Default: 4.
        kernel_sizes (list[int]): DWConv kernel sizes per expert.
            Default: [3, 5, 7, 9].
        lce_kernel (int): LCE strip conv kernel size. Default: 11.
        top_k (int): Top-k experts per grid cell. Default: 3.
        init_temperature (float): Initial router temperature. Default: 1.0.
    """

    def __init__(self, channels, num_experts=4, kernel_sizes=None,
                 lce_kernel=11, top_k=3, init_temperature=1.0):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        if kernel_sizes is None:
            kernel_sizes = [(3 + 2 * i) for i in range(num_experts)]
        self.kernel_sizes = kernel_sizes

        # ---- LCE Branch (paper: AvgPool → DWConv 1×K → DWConv K×1 → 1×1 Conv) ----
        self.lce_avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.lce_strip_h = nn.Conv2d(channels, channels, (1, lce_kernel),
                                     padding=(0, lce_kernel // 2),
                                     groups=channels, bias=False)
        self.lce_strip_w = nn.Conv2d(channels, channels, (lce_kernel, 1),
                                     padding=(lce_kernel // 2, 0),
                                     groups=channels, bias=False)
        self.lce_proj = nn.Conv2d(channels, channels, 1, bias=False)

        # ---- MFE Branch ----
        # Paper: experts differ in kernel sizes AND channel counts
        # (small kernel → more channels for fine details;
        #  large kernel → fewer channels for spatial context)
        expert_channels = self._compute_expert_channels(channels, num_experts)

        self.expert_reduce = nn.ModuleList()
        self.expert_dwconv = nn.ModuleList()
        self.expert_coord_att = nn.ModuleList()
        self.expert_expand = nn.ModuleList()

        for i in range(num_experts):
            ec = expert_channels[i]
            k = kernel_sizes[i]
            # pointwise reduce: C → ec  (paper: depth-wise separable conv)
            self.expert_reduce.append(
                nn.Conv2d(channels, ec, 1, bias=False))
            # DWConv k×k  (paper Eq.10, no BN, no activation)
            self.expert_dwconv.append(
                nn.Conv2d(ec, ec, k, padding=k // 2, groups=ec, bias=False))
            # Coordinate Attention  (paper Eq.11-16, no BN)
            self.expert_coord_att.append(CoordAttention(ec))
            # pointwise expand: ec → C
            self.expert_expand.append(
                nn.Conv2d(ec, channels, 1, bias=False))

        # ---- Grid Router (paper Eq.8: cosine similarity + Top-k) ----
        router_dim = channels // 2
        self.router_proj = nn.Conv2d(channels, router_dim, 1, bias=False)
        self.expert_embedding = nn.Parameter(
            torch.empty(router_dim, num_experts))
        nn.init.orthogonal_(self.expert_embedding)
        self.temperature = nn.Parameter(torch.tensor(init_temperature))

        # Register balance loss buffer
        self.register_buffer('balance_loss', torch.tensor(0.0))

    @staticmethod
    def _compute_expert_channels(total_ch, num_exp):
        """Compute per-expert channel counts.

        Paper: "for experts with smaller kernels, we increase the number
        of channels … for experts with larger kernels, we reduce the
        channel count."  Expert 0 has smallest kernel → most channels.
        """
        if num_exp == 4:
            ratios = [0.40625, 0.3125, 0.1875, 0.09375]
        else:
            raw = [(num_exp - i) for i in range(num_exp)]
            s = float(sum(raw))
            ratios = [r / s for r in raw]
        chs = [max(8, int(total_ch * r + 0.5)) for r in ratios]
        diff = total_ch - sum(chs)
        chs[0] += diff
        return chs

    def forward(self, x):
        B, C, H, W = x.shape

        # ---- LCE: AvgPool → DWConv 1×K → DWConv K×1 → 1×1 Conv ----
        lce_out = self.lce_avgpool(x)
        lce_out = self.lce_strip_h(lce_out)
        lce_out = self.lce_strip_w(lce_out)
        lce_out = self.lce_proj(lce_out)

        # ---- Grid Router ----
        temp = self.temperature.clamp(0.5, 1.5)
        proj = self.router_proj(x)                               # [B, D, H, W]

        proj_flat = proj.view(B, -1, H * W)                      # [B, D, HW]
        proj_norm = F.normalize(proj_flat, dim=1)
        emb_norm = F.normalize(self.expert_embedding, dim=0)     # [D, N]

        sim = torch.bmm(
            proj_norm.transpose(1, 2),                            # [B, HW, D]
            emb_norm.unsqueeze(0).expand(B, -1, -1)              # [B, D, N]
        )                                                         # [B, HW, N]
        sim = sim.permute(0, 2, 1).view(B, -1, H, W)            # [B, N, H, W]

        router_logits = sim / temp
        router_prob = F.softmax(router_logits, dim=1)            # [B, N, H, W]

        # Top-k sparse selection with renormalization (paper Eq.8)
        topk_vals, topk_idx = torch.topk(router_prob, k=self.top_k, dim=1)
        mask = torch.zeros_like(router_prob)
        mask.scatter_(1, topk_idx, topk_vals)
        router_weights = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)

        # Balance Loss: CV² (paper Eq.19)
        if self.training:
            p_mean = router_prob.mean(dim=[0, 2, 3])             # [N]
            cv = p_mean.std() / (p_mean.mean() + 1e-8)
            self.balance_loss = cv ** 2

        # ---- MFE: scale-specialized experts (paper Eq.9-10) ----
        mfe_out = 0
        for i in range(self.num_experts):
            feat = self.expert_reduce[i](x)           # C → ec_i
            feat = self.expert_dwconv[i](feat)        # DWConv k_i×k_i
            feat = self.expert_coord_att[i](feat)     # CoordAttention
            feat = self.expert_expand[i](feat)        # ec_i → C
            weight = router_weights[:, i:i + 1, :, :]  # [B, 1, H, W]
            mfe_out = mfe_out + weight * feat

        # ---- Residual Output (paper Eq.9) ----
        # y = LCE(x) + Σ router_i·MFE_i(x) + x
        return lce_out + mfe_out + x


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """FPN enhanced with ARFC-Block from the ARFN-MoE paper.

    Standard FPN top-down pathway with optional ARFC enhancement at
    each fusion step.

    Args:
        in_channels (list[int]): Input channels per scale.
        out_channels (int): Output channels.
        num_outs (int): Number of output scales.
        fusion_modes (list[str]): ``'add'`` (standard FPN) or
            ``'arfc'`` (FPN addition + ARFCBlock).
        arfc_num_experts (int): MFE experts. Default: 4.
        arfc_top_k (int): Top-k routing. Default: 3.
        arfc_lce_kernel (int): LCE strip kernel size. Default: 11.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes: List[str],
                 arfc_num_experts=4,
                 arfc_top_k=3,
                 arfc_lce_kernel=11,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        self.fusion_modes = fusion_modes
        self.arfc_blocks = nn.ModuleList()

        for mode in fusion_modes:
            if mode == 'arfc':
                self.arfc_blocks.append(
                    ARFCBlock(
                        channels=out_channels,
                        num_experts=arfc_num_experts,
                        top_k=arfc_top_k,
                        lce_kernel=arfc_lce_kernel,
                    ))
            else:
                self.arfc_blocks.append(None)

    @auto_fp16()
    def forward(self, inputs):
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            fusion_idx = used_backbone_levels - 1 - i
            mode = self.fusion_modes[fusion_idx]

            if 'scale_factor' in self.upsample_cfg:
                up_high = F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                up_high = F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

            if mode == 'add':
                # Standard FPN: lateral + upsampled
                laterals[i - 1] = laterals[i - 1] + up_high

            elif mode == 'arfc':
                # FPN addition + ARFC enhancement
                laterals[i - 1] = laterals[i - 1] + up_high
                laterals[i - 1] = self.arfc_blocks[fusion_idx](
                    laterals[i - 1])

            else:
                raise ValueError(f"Unknown fusion mode: {mode}")

        outs = [self.fpn_convs[i](laterals[i])
                for i in range(used_backbone_levels)]

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
                outs.append(
                    self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)

    def get_balance_loss(self):
        """Return accumulated auxiliary balance loss from ARFC blocks."""
        loss = 0.0
        for block in self.arfc_blocks:
            if block is not None:
                loss = loss + block.balance_loss
        return loss
