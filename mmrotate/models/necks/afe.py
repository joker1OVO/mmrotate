import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List, Optional


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from ..builder import ROTATED_NECKS

class AngleFreqEnhance(nn.Module):
    """
    全频主轴与垂直主轴增强模块
    动态检测主频方向，同时增强该方向及其垂直方向，不区分高低频。
    """
    def __init__(self,
                 in_channels: int = 256,
                 k_peaks: int = 1,                # 取前k个主方向（通常1）
                 angle_bandwidth: float = 15.0,   # 角度带宽（度）
                 enhance_alpha: float = 1.2,      # 增强倍数
                 enhance_perp: bool = True,       # 是否增强垂直方向
                 residual: bool = True,
                 c_mid: int = 16,
                 eps: float = 1e-8):
        super().__init__()
        self.k_peaks = k_peaks
        self.angle_bandwidth = angle_bandwidth
        self.enhance_alpha = enhance_alpha
        self.enhance_perp = enhance_perp
        self.residual = residual
        self.c_mid = c_mid
        self.eps = eps

        # 通道压缩与恢复
        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)

        # 预计算极坐标网格（缓存）
        self.register_buffer('angle_grid', None)
        self.register_buffer('radius_grid', None)
        self.r_max = None

    def _compute_grid(self, H: int, W: int, device: torch.device):
        """计算极坐标角度和半径网格（中心化频谱）"""
        if self.angle_grid is None or self.angle_grid.shape[-2:] != (H, W):
            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(torch.arange(H, device=device),
                                  torch.arange(W, device=device), indexing='ij')
            r = torch.sqrt((y - cy)**2 + (x - cx)**2)
            theta = torch.atan2(y - cy, x - cx) + math.pi   # [0, 2π)
            self.angle_grid = theta
            self.radius_grid = r
            self.r_max = min(cy, cx)

    def _get_peak_directions(self, mag: torch.Tensor):
        """
        从幅度谱中检测主频方向（弧度，范围[0,π)）
        mag: [B, C_mid, H, W] 幅度谱（已fftshift）
        返回: [B, k_peaks] 主方向弧度
        """
        B, C, H, W = mag.shape
        device = mag.device
        theta = self.angle_grid % math.pi   # [H, W]
        r = self.radius_grid

        # 沿通道平均，得到 [B, H, W]
        mag_avg = mag.mean(dim=1)

        # 全频加权（不设高频阈值，但可以忽略零频）
        zero_freq_mask = (r == 0).float().unsqueeze(0)  # [1, H, W]
        weighted_mag = mag_avg * (1 - zero_freq_mask)   # 排除零频

        n_bins = 180
        bin_edges = torch.linspace(0, math.pi, n_bins+1, device=device)
        theta_flat = theta.reshape(-1)
        weighted_flat = weighted_mag.reshape(B, -1)

        energy_per_bin = torch.zeros(B, n_bins, device=device)
        for b in range(B):
            bin_idx = torch.bucketize(theta_flat, bin_edges) - 1
            bin_idx = bin_idx.clamp(0, n_bins-1)
            energy = weighted_flat[b]
            energy_per_bin[b] = torch.zeros(n_bins, device=device).scatter_add_(0, bin_idx, energy)

        # 平滑
        kernel = torch.tensor([0.25, 0.5, 0.25], device=device).view(1,1,3)
        e_smooth = F.conv1d(energy_per_bin.unsqueeze(1), kernel, padding=1).squeeze(1)

        peaks = []
        for b in range(B):
            e = e_smooth[b]
            # 局部极大值检测
            left = torch.roll(e, 1)
            right = torch.roll(e, -1)
            is_peak = (e > left) & (e > right)
            peak_indices = torch.where(is_peak)[0]
            if len(peak_indices) == 0:
                max_idx = torch.argmax(e)
                peak_indices = torch.tensor([max_idx], device=device)
            else:
                peak_vals = e[peak_indices]
                order = torch.argsort(peak_vals, descending=True)
                peak_indices = peak_indices[order[:self.k_peaks]]
            bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2
            peak_angles = bin_center[peak_indices]
            if len(peak_angles) < self.k_peaks:
                last = peak_angles[-1] if len(peak_angles) > 0 else torch.tensor(0.0, device=device)
                peak_angles = torch.cat([peak_angles, last.repeat(self.k_peaks - len(peak_angles))])
            peaks.append(peak_angles)
        return torch.stack(peaks, dim=0)   # [B, k_peaks]

    def _create_gain_map(self, mag_shape, peak_angles):
        """
        生成增益图：主方向和垂直方向增强，其他区域不变（增益=1）
        mag_shape: (B, C_mid, H, W)
        peak_angles: [B, k_peaks] 弧度
        """
        B, C, H, W = mag_shape
        device = peak_angles.device
        theta = self.angle_grid          # [H, W]
        bw = math.radians(self.angle_bandwidth)

        gain = torch.ones(B, C, H, W, device=device)

        for b in range(B):
            peaks = peak_angles[b]
            for p_angle in peaks:
                # 主方向增强
                angle_diff = torch.abs(theta - p_angle)
                angle_diff = torch.min(angle_diff, math.pi - angle_diff)
                in_main = (angle_diff <= bw)
                gain[b, :, in_main] = self.enhance_alpha

                # 垂直方向增强
                if self.enhance_perp:
                    perp_angle = (p_angle + math.pi/2) % math.pi
                    angle_diff_perp = torch.abs(theta - perp_angle)
                    angle_diff_perp = torch.min(angle_diff_perp, math.pi - angle_diff_perp)
                    in_perp = (angle_diff_perp <= bw)
                    gain[b, :, in_perp] = self.enhance_alpha

        # 零频保持不变（增益=1）
        zero_mask = (self.radius_grid == 0)
        gain[:, :, zero_mask] = 1.0
        return gain

    @auto_fp16()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        x_proj = self.proj_in(x)                # [B, c_mid, H, W]
        self._compute_grid(H, W, device)

        # FFT
        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs()
        phase = x_fft_shift.angle()

        # 检测主方向
        peak_angles = self._get_peak_directions(mag)   # [B, k_peaks]

        # 生成增益图
        gain = self._create_gain_map(mag.shape, peak_angles)

        # 应用增益（幅度增强，相位不变）
        x_fft_shift_enhanced = x_fft_shift * gain.to(dtype=x_fft_shift.dtype)

        # IFFT
        x_fft_ishift = torch.fft.ifftshift(x_fft_shift_enhanced, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real

        # 恢复通道
        x_enh = self.proj_out(x_enh)

        if self.residual:
            return x + x_enh
        else:
            return x_enh


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """
    创新点：在 FPN 的侧向连接处对 P2-P5 分别进行频域角度增强，随后进行 Top-down 融合。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 # 控制哪些层需要 AFE，默认 [0,1,2,3] 对应 P2, P3, P4, P5
                 enhance_levels: List[int] = [0, 1, 2, 3],
                 afe_cfg: dict = dict(n_angles=32, c_mid=16),
                 **kwargs):
        super(AngleFreqEnhanceFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        self.enhance_levels = enhance_levels

        # 为每个 backbone 级别构建独立的 AFE 模块
        self.afe_modules = nn.ModuleList()
        for i in range(len(in_channels)):
            if i in enhance_levels:
                # 传入 out_channels，因为 AFE 作用在侧向连接（256通道）之后
                self.afe_modules.append(AngleFreqEnhance(in_channels=out_channels, **afe_cfg))
            else:
                self.afe_modules.append(nn.Identity())

    @auto_fp16()
    def forward(self, inputs):
        # 1. 构建侧向连接并立即应用 AFE 增强
        # laterals[0]->P2, laterals[1]->P3, laterals[2]->P4, laterals[3]->P5
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            feat = lateral_conv(inputs[i + self.start_level])
            # 对当前层进行 AFE 增强（P2-P5 独立处理）
            feat = self.afe_modules[i](feat)
            laterals.append(feat)

        # 2. Top-down 融合
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # 这里的融合逻辑：上一层上采样 + 当前层(已增强)
            prev_shape = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 3. 构建输出层 (P2-P5 的 3x3 卷积)
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # 4. 额外的层 (如 P6, P7)
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