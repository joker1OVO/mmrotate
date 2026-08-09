import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List


# ============================ 基础模块 ============================

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
        self.fuse_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv_h = nn.Conv2d(channels, channels, 1)
        self.conv_w = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        x_h = x.mean(dim=2, keepdim=True)       # [B, C, 1, W]
        x_w = x.mean(dim=3, keepdim=True)       # [B, C, H, 1]
        x_w = x_w.permute(0, 1, 3, 2)            # [B, C, 1, H]

        cat = torch.cat([x_h, x_w], dim=3)       # [B, C, 1, W+H]
        fused = self.fuse_conv(cat)               # [B, C, 1, W+H]

        h_feat = fused[:, :, :, :W]               # [B, C, 1, W]
        w_feat = fused[:, :, :, W:]               # [B, C, 1, H]

        ca_h = self.conv_h(h_feat).sigmoid()      # [B, C, 1, W]
        ca_w = self.conv_w(w_feat).sigmoid()      # [B, C, 1, H]

        return x * ca_h * ca_w.permute(0, 1, 3, 2)


# ============================ 旋转工具 ============================

def _estimate_principal_angle(x, angle_pool_size=64, mirror_pad=8, eps=1e-8):
    """Estimate the principal *spatial edge* direction via FFT + circular statistics.

    Returns:
        [B] tensor of rotation angles in radians (CCW-positive).
        Rotating by this angle aligns the dominant edge with horizontal.
    """
    B, C, H, W = x.shape
    device = x.device

    x_mean = x.mean(dim=1, keepdim=True)                     # [B, 1, H, W]
    ps = min(angle_pool_size, H, W)
    x_pool = F.adaptive_avg_pool2d(x_mean, (ps, ps))        # [B, 1, ps, ps]

    pad = mirror_pad
    x_pad = F.pad(x_pool, (pad, pad, pad, pad), mode='reflect')

    x_fft = torch.fft.fft2(x_pad)
    x_fft_s = torch.fft.fftshift(x_fft, dim=(-2, -1))
    mag = x_fft_s.abs() + eps                               # [B, 1, Hf, Wf]

    Hf, Wf = mag.shape[-2], mag.shape[-1]
    fy = torch.arange(Hf, device=device, dtype=torch.float32) - Hf / 2.0
    fx = torch.arange(Wf, device=device, dtype=torch.float32) - Wf / 2.0
    fy_g, fx_g = torch.meshgrid(fy, fx, indexing='ij')

    radius = torch.sqrt(fy_g ** 2 + fx_g ** 2)
    freq_angle = torch.atan2(fy_g, fx_g)

    mask = (radius > 1.0) & (radius < max(Hf, Wf) * 0.4)
    weighted = mag[:, 0] * radius.unsqueeze(0) * mask.unsqueeze(0)

    cos2 = torch.cos(2.0 * freq_angle)
    sin2 = torch.sin(2.0 * freq_angle)
    w_cos = (weighted * cos2).sum(dim=(-2, -1))
    w_sin = (weighted * sin2).sum(dim=(-2, -1))
    freq_dir = 0.5 * torch.atan2(w_sin, w_cos)               # [B]

    spatial_edge = freq_dir + math.pi / 2.0
    return -spatial_edge


def _rotate(x, angles):
    """Rotate feature map around centre by per-sample angles (CCW positive).

    Args:
        x: [B, C, H, W]
        angles: [B] radians
    Returns:
        [B, C, H, W]
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

    grid = F.affine_grid(theta, torch.Size([B, C, H, W]), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear',
                         padding_mode='zeros', align_corners=False)


# ============================ LCE（原始条带卷积） ============================

class LCE(nn.Module):
    """Local Context Enhancement — horizontal + vertical strip DWConv.

    Original LCE from ARFN-MoE without AvgPool.  This is the simplest
    single-branch spatial strip convolution operating on the original
    (unrotated, untransformed) feature map.
    """

    def __init__(self, channels, strip_kernel=11):
        super().__init__()
        self.strip_h = nn.Conv2d(channels, channels, (1, strip_kernel),
                                 padding=(0, strip_kernel // 2),
                                 groups=channels, bias=False)
        self.strip_w = nn.Conv2d(channels, channels, (strip_kernel, 1),
                                 padding=(strip_kernel // 2, 0),
                                 groups=channels, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        out = self.strip_h(x)
        out = self.strip_w(out)
        return self.proj(out)


# ============================ A²RC（角度自适应旋转卷积） ============================

class A2RC(nn.Module):
    """Angle-Adaptive Rotation Convolution.

    Single-branch: estimates principal edge direction via FFT →
    rotates feature map → applies strip conv → rotates back.

    This is the core rotation-adaptive branch that aligns strip convolutions
    with the dominant edge orientation in each sample.
    """

    def __init__(self, channels, strip_kernel=11,
                 angle_pool_size=64, mirror_pad=8):
        super().__init__()
        self.angle_pool_size = angle_pool_size
        self.mirror_pad = mirror_pad

        self.strip_h = nn.Conv2d(channels, channels, (1, strip_kernel),
                                 padding=(0, strip_kernel // 2),
                                 groups=channels, bias=False)
        self.strip_w = nn.Conv2d(channels, channels, (strip_kernel, 1),
                                 padding=(strip_kernel // 2, 0),
                                 groups=channels, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        B = x.shape[0]

        # Estimate per-sample rotation angle
        angles = _estimate_principal_angle(
            x, self.angle_pool_size, self.mirror_pad)

        # Rotate → strip conv → rotate back
        x_rot = _rotate(x, angles)
        out = self.strip_h(x_rot)
        out = self.strip_w(out)
        out = _rotate(out, -angles)
        return self.proj(out)


# ============================ Rot45（固定45°旋转条带卷积） ============================

class Rot45Block(nn.Module):
    """Fixed 45° rotation + strip convolution.

    Rotates feature map by a fixed +45° (π/4), applies horizontal + vertical
    strip DWConv, then rotates back.  This captures features aligned with the
    45° diagonal — a common orientation in remote-sensing objects.

    Unlike A²RC which estimates per-sample rotation via FFT, Rot45 uses a
    fixed angle and complements the adaptive branch with a hard-coded prior.
    """

    def __init__(self, channels, strip_kernel=11):
        super().__init__()
        self.strip_h = nn.Conv2d(channels, channels, (1, strip_kernel),
                                 padding=(0, strip_kernel // 2),
                                 groups=channels, bias=False)
        self.strip_w = nn.Conv2d(channels, channels, (strip_kernel, 1),
                                 padding=(strip_kernel // 2, 0),
                                 groups=channels, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        B = x.shape[0]
        angle = math.pi / 4.0               # fixed +45°
        angles = torch.full((B,), angle, device=x.device)

        x_rot = _rotate(x, angles)
        out = self.strip_h(x_rot)
        out = self.strip_w(out)
        out = _rotate(out, -angles)
        return self.proj(out)


# ============================ FTB（频域双分支） ============================

class FreqDualBranch(nn.Module):
    """Frequency-domain Dual Branch: H→W Sequential Strip + 3×3 DWConv.

    .. code-block:: text

        x → rFFT → [Re, Im] ─┬─ H-Strip(1×K) → W-Strip(K×1) → irFFT ─┐
                              └─ DWConv(3×3) → irFFT ─────────────────┤
                                                                       │
                              spatial-domain sum ←─────────────────────┘
                                  │
                               1×1 Proj

    Two parallel branches on the same rFFT result, each with its own irFFT,
    fused by element-wise sum in the spatial domain.
    The H-W branch applies horizontal then vertical strip convolutions
    sequentially, while the square branch applies 2D local frequency mixing.
    Each branch processes Re and Im with separate (unshared) weights.

    Args:
        channels (int): Feature channels.
        freq_strip_kernel (int): Kernel size for H/W strip convs. Default: 3.
        freq_2d_kernel (int): Kernel size for 2D DWConv. Default: 3.
    """

    def __init__(self, channels, freq_strip_kernel=3, freq_2d_kernel=3):
        super().__init__()

        # ---- H-Strip (沿频率 u 方向, 1×K) ----
        self.h_re = nn.Conv2d(channels, channels, (1, freq_strip_kernel),
                              padding=(0, freq_strip_kernel // 2),
                              groups=channels, bias=False)
        self.h_im = nn.Conv2d(channels, channels, (1, freq_strip_kernel),
                              padding=(0, freq_strip_kernel // 2),
                              groups=channels, bias=False)

        # ---- W-Strip (沿频率 v 方向, K×1) ----
        self.w_re = nn.Conv2d(channels, channels, (freq_strip_kernel, 1),
                              padding=(freq_strip_kernel // 2, 0),
                              groups=channels, bias=False)
        self.w_im = nn.Conv2d(channels, channels, (freq_strip_kernel, 1),
                              padding=(freq_strip_kernel // 2, 0),
                              groups=channels, bias=False)

        # ---- 3×3 2D DWConv ----
        self.conv2d_re = nn.Conv2d(channels, channels, freq_2d_kernel,
                                   padding=freq_2d_kernel // 2,
                                   groups=channels, bias=False)
        self.conv2d_im = nn.Conv2d(channels, channels, freq_2d_kernel,
                                   padding=freq_2d_kernel // 2,
                                   groups=channels, bias=False)

        # ---- 1×1 Proj after irFFT ----
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        x_fft = torch.fft.rfft2(x)                     # [B, C, H, W//2+1]
        x_re, x_im = x_fft.real, x_fft.imag

        # Two parallel branches — each gets its own irFFT, fused in spatial domain:
        #   ① H→W sequential strip (水平-垂直) → irFFT
        #   ② 3×3 DWConv (方形) → irFFT
        # Each with separate Re/Im weights

        # Branch ①: H→W strip
        re_strip = self.w_re(self.h_re(x_re))
        im_strip = self.w_im(self.h_im(x_im))
        out_strip = torch.fft.irfft2(torch.complex(re_strip, im_strip),
                                     s=(x.shape[2], x.shape[3]))

        # Branch ②: 3×3 DWConv
        re_2d = self.conv2d_re(x_re)
        im_2d = self.conv2d_im(x_im)
        out_2d = torch.fft.irfft2(torch.complex(re_2d, im_2d),
                                  s=(x.shape[2], x.shape[3]))

        # Fuse in spatial domain
        return self.proj(out_strip + out_2d)


# ============================ ARFCBlock（多分支总模块） ============================

class ARFCBlock(nn.Module):
    """Adaptive Receptive Field Convolution Block.

    .. code-block:: text

        y = x                          ← residual (identity skip)
          + A²RC(x)                    ← 旋转自适应卷积 (LCE+A²RC+Rot45 → 1×1 merge)
          + FTB(x)                     ← 频域双分支 (各自irFFT → 空间域相加)
          + LargeKernel(x)             ← 大核卷积 (DWConv 5×5+7×7+9×9, 相加)


    - **大核卷积分支 (Large Kernel Convolution)** : 3 个并行的深度可分离大核卷积,
      kernel_size=[5,7,9] (DWConv, groups=channels), 各自独立权重, 通道不变(256), 输出按元素相加.

    - **A²RC（旋转自适应卷积）** : 三个子分支共享输入、各自处理、相加后 1×1 融合.
      ① LCE — H+W strip DWConv (1×11 / 11×1) on original feature.
      ② Angle-Adaptive — FFT-estimated rotation + strip conv + rotate back.
      ③ Rot45 — fixed +45° rotation + strip conv + rotate back.

    - **FTB（频域双分支）** :
      rFFT → [Re, Im] → two parallel branches, each with its own irFFT, fused in spatial domain:
      ① H-Strip(1×K) → W-Strip(K×1) → irFFT,
      ② DWConv(3×3) → irFFT.
      → element-wise sum → 1×1 Proj.
      Re/Im use independent (unshared) weights per branch.

    - **Residual** : identity skip connection ``+ x``.
    """

    def __init__(self, channels, large_kernels=None,
                 strip_kernel=11, freq_kernel=3,
                 enable_lce=True, enable_a2rc=True,
                 enable_fdsc=True, enable_large_kernel=True,
                 enable_rot45=True):
        super().__init__()
        self.channels = channels
        self.enable_lce = enable_lce
        self.enable_a2rc = enable_a2rc
        self.enable_rot45 = enable_rot45
        self.enable_fdsc = enable_fdsc
        self.enable_large_kernel = enable_large_kernel

        # ==================== A²RC（旋转自适应卷积，三分支 → 1×1 merge） ====================
        # ① LCE: 水平+垂直条带卷积 (原始特征, 不旋转)
        if enable_lce:
            self.lce = LCE(channels, strip_kernel)

        # ② Angle-Adaptive: FFT自适应旋转 + strip conv
        if enable_a2rc:
            self.a2rc = A2RC(channels, strip_kernel=strip_kernel)

        # ③ Rot45: 固定45°旋转 + strip conv
        if enable_rot45:
            self.rot45 = Rot45Block(channels, strip_kernel=strip_kernel)

        # 旋转分支 merge: 当 2+ 子分支激活时使用 1×1 融合
        rot_active = sum([enable_lce, enable_a2rc, enable_rot45])
        if rot_active >= 2:
            self.rot_merge = nn.Conv2d(channels, channels, 1, bias=False)
        else:
            self.rot_merge = None

        # ==================== 频域双分支: FTB = H→W-Strip + 3×3 DWConv ====================
        if enable_fdsc:
            self.ftb = FreqDualBranch(channels,
                                        freq_strip_kernel=freq_kernel,
                                        freq_2d_kernel=3)

        # ==================== 大核卷积分支: 3 个深度可分离大核卷积 (k=5,7,9) ====================
        if enable_large_kernel:
            if large_kernels is None:
                large_kernels = [5, 7, 9]
            self.large_kernel_convs = nn.ModuleList([
                nn.Conv2d(channels, channels, k, padding=k // 2,
                          groups=channels, bias=False)
                for k in large_kernels
            ])

    def forward(self, x):
        # ---- Residual (identity) ----
        identity = x

        # ==================== A²RC（旋转自适应卷积，三分支 → 1×1 merge） ====================
        rotation_out = 0
        # ① LCE: 水平+垂直条带 (原始特征, 不旋转)
        if self.enable_lce:
            rotation_out = rotation_out + self.lce(x)
        # ② Angle-Adaptive: FFT自适应旋转 + strip conv
        if self.enable_a2rc:
            rotation_out = rotation_out + self.a2rc(x)
        # ③ Rot45: 固定45°旋转 + strip conv
        if self.enable_rot45:
            rotation_out = rotation_out + self.rot45(x)

        # 1×1 merge (3 sub-branches → 1)
        if self.rot_merge is not None:
            rotation_out = self.rot_merge(rotation_out)

        # ==================== FTB（频域双分支） ====================
        freq_out = 0
        if self.enable_fdsc:
            freq_out = self.ftb(x)

        # ==================== Large Kernel（大核卷积分支，k=5,7,9） ====================
        large_kernel_out = 0
        if self.enable_large_kernel:
            for conv in self.large_kernel_convs:
                large_kernel_out = large_kernel_out + conv(x)

        # ==================== 最终求和 ====================
        # y = x + A²RC(x) + FTB(x) + LargeKernel(x)
        return identity + large_kernel_out + rotation_out + freq_out


# ============================ SAFEFPN ============================

@ROTATED_NECKS.register_module()
class SAFEFPN(FPN):
    """SAFEFPN: FPN enhanced with SAFE Blocks.

    Standard FPN top-down pathway with optional SAFE enhancement at
    each fusion step.

    Each SAFE (ARFCBlock) contains:
      - A²RC（旋转自适应卷积）: LCE + Angle-Adaptive + Rot45 三个子分支 → 1×1 merge.
      - FTB（频域双分支）: rFFT → 两路各自irFFT → 空间域相加.
      - Large Kernel（大核卷积）: 3 个并行 DWConv (k=5,7,9), 相加.

    Args:
        in_channels (list[int]): Input channels per scale.
        out_channels (int): Output channels.
        num_outs (int): Number of output scales.
        fusion_modes (list[str]): ``'add'`` or ``'safe'``.
        safe_strip_kernel (int): Strip conv kernel size. Default: 11.
        safe_freq_kernel (int): FTB kernel size. Default: 3.
        safe_large_kernels (list[int]): Large kernel sizes. Default: [5, 7, 9].
        safe_enable_lce (bool): A²RC 子分支 LCE. Default: True.
        safe_enable_a2rc (bool): A²RC 子分支 Angle-Adaptive. Default: True.
        safe_enable_rot45 (bool): A²RC 子分支 Rot45. Default: True.
        safe_enable_fdsc (bool): FTB 频域双分支. Default: True.
        safe_enable_large_kernel (bool): Large Kernel 大核卷积分支. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes: List[str],
                 safe_strip_kernel=11,
                 safe_freq_kernel=3,
                 safe_large_kernels=None,
                 safe_enable_lce=True,
                 safe_enable_a2rc=True,
                 safe_enable_rot45=True,
                 safe_enable_fdsc=True,
                 safe_enable_large_kernel=True,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        self.fusion_modes = fusion_modes
        self.safe_blocks = nn.ModuleList()

        for mode in fusion_modes:
            if mode == 'safe':
                self.safe_blocks.append(
                    ARFCBlock(
                        channels=out_channels,
                        strip_kernel=safe_strip_kernel,
                        freq_kernel=safe_freq_kernel,
                        large_kernels=safe_large_kernels,
                        enable_lce=safe_enable_lce,
                        enable_a2rc=safe_enable_a2rc,
                        enable_rot45=safe_enable_rot45,
                        enable_fdsc=safe_enable_fdsc,
                        enable_large_kernel=safe_enable_large_kernel,
                    ))
            else:
                self.safe_blocks.append(None)

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
                laterals[i - 1] = laterals[i - 1] + up_high

            elif mode == 'safe':
                laterals[i - 1] = laterals[i - 1] + up_high
                laterals[i - 1] = self.safe_blocks[fusion_idx](
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

