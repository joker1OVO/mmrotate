import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List, Optional

class AngleFreqEnhance(nn.Module):
    """
    频域角度增强模块（支持动态主频检测，分策略增强/削弱）
    """
    def __init__(self,
                 in_channels: int = 256,
                 strategy: str = 'small',   # 'small' or 'large'
                 k_peaks: int = 2,
                 angle_bandwidth: float = 15.0,
                 high_freq_ratio: float = 0.3,
                 low_freq_ratio: float = 0.2,
                 enhance_alpha: float = 1.2,
                 suppress_beta: float = 0.8,
                 residual: bool = True,
                 c_mid: int = 16,
                 eps: float = 1e-8):
        super().__init__()
        self.strategy = strategy
        self.k_peaks = k_peaks
        self.angle_bandwidth = angle_bandwidth
        self.high_freq_ratio = high_freq_ratio
        self.low_freq_ratio = low_freq_ratio
        self.enhance_alpha = enhance_alpha
        self.suppress_beta = suppress_beta
        self.residual = residual
        self.c_mid = c_mid
        self.eps = eps

        # 通道压缩与恢复
        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)

        # 预计算掩码的占位符（极坐标网格）
        self.register_buffer('angle_grid', None)
        self.register_buffer('radius_grid', None)
        self.r_max = None  # 普通属性

    def _compute_grid(self, H: int, W: int, device: torch.device):
        """计算极坐标网格（角度和半径），缓存"""
        if self.angle_grid is None or self.angle_grid.shape[-2:] != (H, W):
            cy, cx = H // 2, W // 2
            y, x = torch.meshgrid(torch.arange(H, device=device),
                                  torch.arange(W, device=device), indexing='ij')
            r = torch.sqrt((y - cy)**2 + (x - cx)**2)
            theta = torch.atan2(y - cy, x - cx) + math.pi  # [0, 2π)
            self.angle_grid = theta
            self.radius_grid = r
            self.r_max = min(cy, cx)

    def _get_peak_directions(self, mag: torch.Tensor):
        """
        从幅度谱中检测主频方向（角度，弧度）。
        mag: [B, C_mid, H, W] 幅度谱（已 fftshift）
        返回: [B, k_peaks] 每个样本的主频方向（弧度，范围[0,π)）
        """
        B, C, H, W = mag.shape
        device = mag.device
        theta = self.angle_grid % math.pi   # [H, W]
        r = self.radius_grid
        high_mask = (r > self.high_freq_ratio * self.r_max).float()
        mag_avg = mag.mean(dim=1)  # [B,H,W]
        weighted_mag = mag_avg * high_mask  # [B,H,W]

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
        e_smooth = torch.nn.functional.conv1d(energy_per_bin.unsqueeze(1), kernel, padding=1).squeeze(1)

        peaks = []
        for b in range(B):
            e = e_smooth[b]
            mean_e = e.mean()
            # 找局部极大值
            left = torch.roll(e, 1)
            right = torch.roll(e, -1)
            is_peak = (e > mean_e) & (e > left) & (e > right)
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
                # 补全最后一个值
                last = peak_angles[-1] if len(peak_angles)>0 else torch.tensor(0.0, device=device)
                peak_angles = torch.cat([peak_angles, last.repeat(self.k_peaks - len(peak_angles))])
            peaks.append(peak_angles)
        peaks_tensor = torch.stack(peaks, dim=0)  # [B, k_peaks]
        return peaks_tensor

    def _create_gain_map(self, mag_shape, peak_angles):
        """
        根据策略创建增益图 gain: [B, C_mid, H, W]
        peak_angles: [B, k_peaks] 弧度
        """
        B, C, H, W = mag_shape
        device = peak_angles.device
        theta = self.angle_grid          # [H, W]
        r = self.radius_grid
        bw = math.radians(self.angle_bandwidth)

        # 布尔掩码：高频、低频
        high_mask_bool = (r > self.high_freq_ratio * self.r_max)   # bool
        low_mask_bool = (r <= self.low_freq_ratio * self.r_max) if self.strategy == 'large' else None

        # 初始化增益为1
        gain = torch.ones(B, C, H, W, device=device)

        # 创建一个布尔增强掩码，标记哪些位置被增强
        enhance_mask = torch.zeros(B, C, H, W, dtype=torch.bool, device=device)

        for b in range(B):
            peaks = peak_angles[b]   # [k_peaks]
            for p_angle in peaks:
                if self.strategy == 'small':
                    # 增强主频方向的高频
                    angle_diff = torch.abs(theta - p_angle)
                    angle_diff = torch.min(angle_diff, math.pi - angle_diff)
                    in_region = (angle_diff <= bw)   # bool
                    region_mask = in_region & high_mask_bool
                    # 更新增强掩码
                    enhance_mask[b, :, :, :] = enhance_mask[b, :, :, :] | region_mask
                else:  # 'large'
                    perp_angle = (p_angle + math.pi/2) % math.pi
                    angle_diff = torch.abs(theta - perp_angle)
                    angle_diff = torch.min(angle_diff, math.pi - angle_diff)
                    in_region = (angle_diff <= bw)
                    region_mask = in_region & low_mask_bool
                    enhance_mask[b, :, :, :] = enhance_mask[b, :, :, :] | region_mask

        # 对增强区域应用 enhance_alpha
        gain = torch.where(enhance_mask,
                           torch.tensor(self.enhance_alpha, device=device),
                           gain)

        # 削弱其他方向的高频（高频且不在增强掩码中）
        other_high_mask = high_mask_bool & (~enhance_mask)
        gain = torch.where(other_high_mask,
                           torch.tensor(self.suppress_beta, device=device),
                           gain)

        return gain

    @auto_fp16()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        # 通道压缩
        x_proj = self.proj_in(x)
        # 极坐标网格
        self._compute_grid(H, W, device)
        # FFT
        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs()
        phase = x_fft_shift.angle()
        # 检测主频方向
        peak_angles = self._get_peak_directions(mag)
        # 生成增益图
        gain = self._create_gain_map(mag.shape, peak_angles)
        # 应用增益
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
    FPN with adaptive frequency enhancement for different levels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 enhance_levels: List[int] = [0, 1, 2, 3],
                 small_levels: List[int] = [0, 1],
                 large_levels: List[int] = [2, 3],
                 afe_base_cfg: dict = dict(
                     k_peaks=2,
                     angle_bandwidth=15.0,
                     high_freq_ratio=0.3,
                     low_freq_ratio=0.2,
                     enhance_alpha=1.2,
                     suppress_beta=0.8,
                     c_mid=16,
                 ),
                 **kwargs):
        super(AngleFreqEnhanceFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)
        self.enhance_levels = enhance_levels
        self.small_levels = small_levels
        self.large_levels = large_levels
        self.afe_modules = nn.ModuleList()
        for i in range(len(in_channels)):
            if i in enhance_levels:
                if i in small_levels:
                    cfg = afe_base_cfg.copy()
                    cfg['strategy'] = 'small'
                elif i in large_levels:
                    cfg = afe_base_cfg.copy()
                    cfg['strategy'] = 'large'
                else:
                    self.afe_modules.append(nn.Identity())
                    continue
                self.afe_modules.append(AngleFreqEnhance(in_channels=out_channels, **cfg))
            else:
                self.afe_modules.append(nn.Identity())

    @auto_fp16()
    def forward(self, inputs):
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            feat = lateral_conv(inputs[i + self.start_level])
            feat = self.afe_modules[i](feat)
            laterals.append(feat)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i-1] = laterals[i-1] + upsampled
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
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
                for i in range(used_backbone_levels+1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)