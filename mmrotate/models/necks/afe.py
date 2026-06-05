import math
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


class MultiBranchLCE(nn.Module):
    """Multi-Branch Local Context Enhancement.

    Three branches with **independent** weights, summed and projected:

    1. **Original Strip**: ``1×K`` → ``K×1`` depth-wise strip convs
       (horizontal + vertical), same as the original LCE without AvgPool.

    2. **Adaptive Rotation**: estimates the principal edge direction via
       FFT (mirror-padded to suppress spectral leakage), rotates the
       feature map, applies strip convs, and rotates back.  Angle is
       per-image global (one scalar per sample).

    3. **Frequency Domain**: ``rfft2 → [Re, Im]``, applies strip convs
       (shared weights for Re/Im), then ``irfft2`` back to spatial.

    No internal residual connection — the outer :class:`ARFCBlock`
    provides the ``+ x`` skip.

    Args:
        channels (int): Input/output channels.
        strip_kernel (int): DWConv kernel size for spatial branches
            (1 & 2).  Default: 11.
        freq_kernel (int): DWConv kernel size for frequency branch
            (3).  Use small values (e.g. 3) — each frequency bin already
            represents a global component.  Default: 3.
        angle_pool_size (int): Target size for ``adaptive_avg_pool2d``
            before FFT angle estimation.  Default: 64.
        mirror_pad (int): Mirror-padding width applied before FFT to
            suppress spectral leakage.  Default: 8.
    """

    def __init__(self, channels, strip_kernel=11, freq_kernel=3,
                 angle_pool_size=64, mirror_pad=8):
        super().__init__()
        self.angle_pool_size = angle_pool_size
        self.mirror_pad = mirror_pad

        # ---- Branch 1: original strip ----
        self.b1_h = nn.Conv2d(channels, channels, (1, strip_kernel),
                              padding=(0, strip_kernel // 2),
                              groups=channels, bias=False)
        self.b1_w = nn.Conv2d(channels, channels, (strip_kernel, 1),
                              padding=(strip_kernel // 2, 0),
                              groups=channels, bias=False)
        self.b1_proj = nn.Conv2d(channels, channels, 1, bias=False)

        # ---- Branch 2: adaptive-rotation strip ----
        self.b2_h = nn.Conv2d(channels, channels, (1, strip_kernel),
                              padding=(0, strip_kernel // 2),
                              groups=channels, bias=False)
        self.b2_w = nn.Conv2d(channels, channels, (strip_kernel, 1),
                              padding=(strip_kernel // 2, 0),
                              groups=channels, bias=False)
        self.b2_proj = nn.Conv2d(channels, channels, 1, bias=False)

        # ---- Branch 3: frequency-domain strip (Re/Im share weights) ----
        self.b3_h = nn.Conv2d(channels, channels, (1, freq_kernel),
                              padding=(0, freq_kernel // 2),
                              groups=channels, bias=False)
        self.b3_w = nn.Conv2d(channels, channels, (freq_kernel, 1),
                              padding=(freq_kernel // 2, 0),
                              groups=channels, bias=False)
        self.b3_proj = nn.Conv2d(channels, channels, 1, bias=False)

        # ---- Merge ----
        self.merge = nn.Conv2d(channels, channels, 1, bias=False)

    # -----------------------------------------------------------------
    #  Angle estimation
    # -----------------------------------------------------------------

    def _estimate_principal_angle(self, x):
        """Estimate the principal *spatial edge* direction via FFT.

        Pipeline:  ``mean(C) → adaptive_pool → mirror_pad → FFT →
        magnitude → circular-weighted principal direction``.

        The returned angle is the *rotation* needed to align the
        dominant edge direction with the horizontal axis (so that the
        ``1×K`` strip conv captures the strongest texture).

        Returns:
            ``[B]`` tensor of rotation angles in radians.
        """
        B, C, H, W = x.shape
        device = x.device

        # -- 1. Collapse channels & down-sample --
        x_mean = x.mean(dim=1, keepdim=True)                     # [B, 1, H, W]
        ps = min(self.angle_pool_size, H, W)
        x_pool = F.adaptive_avg_pool2d(x_mean, (ps, ps))        # [B, 1, ps, ps]

        # -- 2. Mirror padding to suppress spectral leakage --
        pad = self.mirror_pad
        x_pad = F.pad(x_pool, (pad, pad, pad, pad), mode='reflect')

        # -- 3. FFT → magnitude spectrum --
        x_fft = torch.fft.fft2(x_pad)
        x_fft_s = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_s.abs() + 1e-8                               # [B, 1, Hf, Wf]

        # -- 4. Frequency coordinate grids --
        Hf, Wf = mag.shape[-2], mag.shape[-1]
        fy = torch.arange(Hf, device=device, dtype=torch.float32) - Hf / 2.0
        fx = torch.arange(Wf, device=device, dtype=torch.float32) - Wf / 2.0
        fy_g, fx_g = torch.meshgrid(fy, fx, indexing='ij')

        radius = torch.sqrt(fy_g ** 2 + fx_g ** 2)
        freq_angle = torch.atan2(fy_g, fx_g)                     # [-π, π]

        # Mask: exclude DC and extreme high frequencies
        mask = (radius > 1.0) & (radius < max(Hf, Wf) * 0.4)

        # Weight by magnitude × radius (edges concentrate at mid-high freq)
        weighted = mag[:, 0] * radius.unsqueeze(0) * mask.unsqueeze(0)

        # -- 5. Double-angle circular statistics (edge dir is π-periodic) --
        cos2 = torch.cos(2.0 * freq_angle)
        sin2 = torch.sin(2.0 * freq_angle)
        w_cos = (weighted * cos2).sum(dim=(-2, -1))
        w_sin = (weighted * sin2).sum(dim=(-2, -1))
        freq_dir = 0.5 * torch.atan2(w_sin, w_cos)               # [B]

        # Spatial edge direction ⊥ frequency direction.
        # Rotate *negatively* so the dominant edge aligns with the
        # horizontal strip axis.
        spatial_edge = freq_dir + math.pi / 2.0
        return -spatial_edge

    # -----------------------------------------------------------------
    #  Spatial rotation (bilinear interpolation)
    # -----------------------------------------------------------------

    @staticmethod
    def _rotate(x, angles):
        """Rotate feature map by per-sample angles.

        Args:
            x: ``[B, C, H, W]``
            angles: ``[B]`` radians, CCW positive.
        Returns:
            ``[B, C, H, W]``
        """
        B, C, H, W = x.shape
        device = x.device

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a

        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        theta[:, 0, 2] = cx - cos_a * cx + sin_a * cy
        theta[:, 1, 2] = cy - sin_a * cx - cos_a * cy

        grid = F.affine_grid(theta, torch.Size([B, C, H, W]),
                             align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=False)

    # -----------------------------------------------------------------
    #  Forward
    # -----------------------------------------------------------------

    def forward(self, x):
        # ---- Branch 1: original strip ----
        b1 = self.b1_h(x)
        b1 = self.b1_w(b1)
        b1 = self.b1_proj(b1)

        # ---- Branch 2: adaptive rotation ----
        angles = self._estimate_principal_angle(x)
        x_rot = self._rotate(x, angles)
        b2 = self.b2_h(x_rot)
        b2 = self.b2_w(b2)
        b2 = self._rotate(b2, -angles)
        b2 = self.b2_proj(b2)

        # ---- Branch 3: frequency domain ----
        x_fft = torch.fft.rfft2(x)                     # [B, C, H, W//2+1]
        x_re, x_im = x_fft.real, x_fft.imag

        re_h = self.b3_h(x_re)
        re_hw = self.b3_w(re_h)
        im_h = self.b3_h(x_im)
        im_hw = self.b3_w(im_h)

        x_freq = torch.complex(re_hw, im_hw)
        b3 = torch.fft.irfft2(x_freq, s=(x.shape[2], x.shape[3]))
        b3 = self.b3_proj(b3)

        # ---- Merge ----
        return self.merge(b1 + b2 + b3)


class ARFCBlock(nn.Module):
    """Adaptive Receptive Field Convolution Block.

    Based on the ARFN-MoE paper (TGRS 2026), with enhanced LCE:

    - **LCE**: :class:`MultiBranchLCE` — three parallel branches
      (original strip, adaptive-rotation strip, frequency-domain strip)
      → summed → ``1×1`` projection.
    - **MFE**:  N experts, each = pointwise reduce → DWConv(k×k) →
      CoordAttention → pointwise expand. Kernel sizes and channel counts
      differ per expert.
    - **Grid Router**: cosine similarity + Top-k sparse selection.
    - **Balance Loss**: CV².
    - **Output**: ``LCE(x) + Σ router_i·MFE_i(x) + x``  (residual, no
      BN/activation).

    Args:
        channels (int): Input/output channels C.
        num_experts (int): Number of MFE experts. Default: 4.
        kernel_sizes (list[int]): DWConv kernel sizes per expert.
            Default: [3, 5, 7, 9].
        lce_kernel (int): Spatial strip conv kernel size (branch 1 & 2).
            Default: 11.
        lce_freq_kernel (int): Frequency-domain strip conv kernel size
            (branch 3).  Default: 3.
        top_k (int): Top-k experts per grid cell. Default: 3.
        init_temperature (float): Initial router temperature. Default: 1.0.
    """

    def __init__(self, channels, num_experts=4, kernel_sizes=None,
                 lce_kernel=11, lce_freq_kernel=3, top_k=3,
                 init_temperature=1.0):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        if kernel_sizes is None:
            kernel_sizes = [(3 + 2 * i) for i in range(num_experts)]
        self.kernel_sizes = kernel_sizes

        # ---- Multi-Branch LCE ---- (replaces the original single-branch LCE)
        self.lce = MultiBranchLCE(
            channels, strip_kernel=lce_kernel,
            freq_kernel=lce_freq_kernel)

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

        # ---- Multi-Branch LCE ----
        lce_out = self.lce(x)

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
                 arfc_lce_freq_kernel=3,
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
                        lce_freq_kernel=arfc_lce_freq_kernel,
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
