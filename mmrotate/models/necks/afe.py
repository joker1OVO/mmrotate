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

    Args:
        in_channels (int): 输入通道数，默认256。
        strategy (str): 'small' 或 'large'。
            - 'small': 增强主频方向的高频，削弱其他方向的高频。
            - 'large': 增强与主频垂直方向的低频，削弱其他方向的高频。
        k_peaks (int): 主频方向数量（取前k个峰值）。
        angle_bandwidth (float): 角度带宽（度数），每个主频方向扩展 ±bandwidth 作为增强区域。
        high_freq_ratio (float): 高频半径阈值（> r_max * ratio 视为高频）。
        low_freq_ratio (float): 低频半径阈值（<= r_max * ratio 视为低频），仅 large 策略使用。
        enhance_alpha (float): 增强系数（主频方向高频 或 垂直方向低频）。
        suppress_beta (float): 削弱系数（其他方向高频）。
        residual (bool): 是否残差连接。
        c_mid (int): 中间通道压缩数。
        eps (float): 稳定项。
    """

    def __init__(self,
                 in_channels: int = 256,
                 strategy: str = 'small',  # 'small' or 'large'
                 k_peaks: int = 2,
                 angle_bandwidth: float = 15.0,  # degrees
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
        self.register_buffer('angle_grid', None)  # [H, W] 角度值（弧度）
        self.register_buffer('radius_grid', None)  # [H, W] 半径值
        self.r_max = None

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
            self.r_max = min(cy, cx)  # 标量 int

    def _get_peak_directions(self, mag: torch.Tensor):
        """
        从幅度谱中检测主频方向（角度，弧度）。
        mag: [B, C_mid, H, W] 幅度谱（已 fftshift）
        返回: [B, k_peaks] 每个样本的主频方向（弧度，范围[0,π)）
        """
        B, C, H, W = mag.shape
        device = mag.device
        # 将角度网格转为 [0, π)（利用对称性，只考虑半圆）
        theta = self.angle_grid % math.pi  # [H, W]
        # 径向加权（强调高频），计算角向能量 E(theta)
        # 简化：对每个角度，累加半径>0.3*r_max的幅度（或加权）
        r = self.radius_grid
        high_mask = (r > self.high_freq_ratio * self.r_max).float()
        # 能量累积：对每个角度，求和幅度 * 径向权重（这里简单用幅度）
        # 为避免循环，使用直方图近似：将角度离散化 bins
        n_bins = 180  # 1度分辨率
        bin_edges = torch.linspace(0, math.pi, n_bins + 1, device=device)
        # 对于每个batch和通道，计算能量分布（取所有通道的平均）
        # 幅度谱形状 [B,C,H,W]，取平均通道
        mag_avg = mag.mean(dim=1)  # [B,H,W]
        # 使用掩码只考虑高频区域
        weighted_mag = mag_avg * high_mask  # [B,H,W]
        # 直方图累积：每个角度bin的能量
        theta_flat = theta.reshape(-1)
        weighted_flat = weighted_mag.reshape(B, -1)  # [B, H*W]
        # 使用 torch.histogramdd 或循环（为简单，逐样本）
        energy_per_bin = torch.zeros(B, n_bins, device=device)
        for b in range(B):
            # 用scatter_add实现直方图
            bin_idx = torch.bucketize(theta_flat, bin_edges) - 1  # [H*W]
            bin_idx = bin_idx.clamp(0, n_bins - 1)
            energy = weighted_flat[b]  # [H*W]
            energy_per_bin[b] = torch.zeros(n_bins, device=device).scatter_add_(0, bin_idx, energy)
        # 平滑（可选，高斯滤波）
        # 找峰值：取能量大于均值的局部极大值
        peaks = []
        for b in range(B):
            e = energy_per_bin[b]
            # 平滑
            kernel = torch.tensor([0.25, 0.5, 0.25], device=device)
            e_smooth = torch.nn.functional.conv1d(e.view(1, 1, -1), kernel.view(1, 1, 3), padding=1).view(-1)
            # 阈值：大于均值
            mean_e = e_smooth.mean()
            # 找局部极大值
            is_peak = (e_smooth > mean_e) & (e_smooth > torch.roll(e_smooth, 1)) & (e_smooth > torch.roll(e_smooth, -1))
            peak_vals = e_smooth[is_peak]
            peak_indices = torch.where(is_peak)[0]
            if len(peak_indices) == 0:
                # 退化为全图最大方向
                max_idx = torch.argmax(e_smooth)
                peak_indices = torch.tensor([max_idx], device=device)
            else:
                # 按能量降序取前k个
                order = torch.argsort(peak_vals, descending=True)
                peak_indices = peak_indices[order[:self.k_peaks]]
            # 将索引转为弧度值
            bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2
            peak_angles = bin_center[peak_indices]  # [k]
            peaks.append(peak_angles)
        # 对齐为 [B, k_peaks]（若不足则重复最后一个）
        max_k = max(len(p) for p in peaks)
        if max_k < self.k_peaks:
            # 补全
            for b in range(B):
                if len(peaks[b]) < self.k_peaks:
                    last = peaks[b][-1] if len(peaks[b]) > 0 else torch.tensor(0.0, device=device)
                    peaks[b] = torch.cat([peaks[b], last.repeat(self.k_peaks - len(peaks[b]))])
        peaks_tensor = torch.stack(peaks, dim=0)  # [B, k_peaks]
        return peaks_tensor

    def _create_gain_map(self, mag_shape, peak_angles):
        """
        根据策略创建增益图 gain: [B, C_mid, H, W]
        peak_angles: [B, k_peaks] 弧度
        """
        B, C, H, W = mag_shape
        device = peak_angles.device
        theta = self.angle_grid  # [H, W]
        r = self.radius_grid
        # 角度带宽（弧度）
        bw = math.radians(self.angle_bandwidth)
        # 初始化增益为1
        gain = torch.ones(B, C, H, W, device=device)
        # 高频掩码
        high_mask = (r > self.high_freq_ratio * self.r_max).float()
        # 低频掩码（用于large策略）
        low_mask = (r <= self.low_freq_ratio * self.r_max).float() if self.strategy == 'large' else None

        # 对于每个batch和每个峰值，生成增强区域
        for b in range(B):
            peaks = peak_angles[b]  # [k_peaks]
            for p_angle in peaks:
                if self.strategy == 'small':
                    # 增强主频方向的高频
                    # 方向区间 [p_angle - bw, p_angle + bw]
                    angle_diff = torch.abs(theta - p_angle)
                    angle_diff = torch.min(angle_diff, math.pi - angle_diff)  # 考虑圆形
                    in_region = (angle_diff <= bw)
                    # 只在高频区域增强
                    region_mask = in_region * high_mask
                    gain[b, :, :, :] = torch.where(region_mask,
                                                   torch.tensor(self.enhance_alpha, device=device),
                                                   gain[b, :, :, :])
                else:  # 'large'
                    # 增强与主频垂直方向的低频
                    perp_angle = (p_angle + math.pi / 2) % math.pi
                    angle_diff = torch.abs(theta - perp_angle)
                    angle_diff = torch.min(angle_diff, math.pi - angle_diff)
                    in_region = (angle_diff <= bw)
                    region_mask = in_region * low_mask
                    gain[b, :, :, :] = torch.where(region_mask,
                                                   torch.tensor(self.enhance_alpha, device=device),
                                                   gain[b, :, :, :])
        # 削弱其他方向的高频（所有不在增强区域的高频点）
        # 注意：上面增强区域已经修改了gain，但我们需要找出所有高频且未被增强的区域
        # 由于增强区域可能多个，我们需要一个总增强掩码
        # 重新计算增强掩码（更清晰的方式：先创建增强掩码，再统一应用）
        # 为避免重复，重新生成一个增强掩码
        enhance_mask = torch.zeros(B, C, H, W, device=device)
        for b in range(B):
            peaks = peak_angles[b]
            for p_angle in peaks:
                if self.strategy == 'small':
                    angle_diff = torch.abs(theta - p_angle)
                    angle_diff = torch.min(angle_diff, math.pi - angle_diff)
                    in_region = (angle_diff <= bw)
                    region_mask = in_region * high_mask
                    enhance_mask[b, :, :, :] = torch.max(enhance_mask[b, :, :, :], region_mask)
                else:
                    perp_angle = (p_angle + math.pi / 2) % math.pi
                    angle_diff = torch.abs(theta - perp_angle)
                    angle_diff = torch.min(angle_diff, math.pi - angle_diff)
                    in_region = (angle_diff <= bw)
                    region_mask = in_region * low_mask
                    enhance_mask[b, :, :, :] = torch.max(enhance_mask[b, :, :, :], region_mask)
        # 其他高频且未被增强的区域，乘以削弱系数
        other_high = high_mask * (1 - enhance_mask)
        gain = gain * (1 - other_high) + other_high * self.suppress_beta
        return gain

    @auto_fp16()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        # 通道压缩
        x_proj = self.proj_in(x)  # [B, c_mid, H, W]
        # 计算极坐标网格（缓存）
        self._compute_grid(H, W, device)
        # FFT
        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs()
        phase = x_fft_shift.angle()
        # 检测主频方向（基于幅度谱，使用平均通道）
        peak_angles = self._get_peak_directions(mag)  # [B, k_peaks]
        # 生成增益图
        gain = self._create_gain_map(mag.shape, peak_angles)  # [B, c_mid, H, W]
        # 应用增益
        x_fft_shift_enhanced = x_fft_shift * gain.to(dtype=x_fft_shift.dtype)
        # IFFT
        x_fft_ishift = torch.fft.ifftshift(x_fft_shift_enhanced, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real
        # 恢复通道
        x_enh = self.proj_out(x_enh)
        # 残差
        if self.residual:
            return x + x_enh
        else:
            return x_enh


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """
    FPN with adaptive frequency enhancement for different levels.
    For P2,P3 (small objects): enhance high-freq in dominant directions.
    For P4,P5 (large objects): enhance low-freq in perpendicular directions.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 # 哪些层应用增强（默认 P2~P5）
                 enhance_levels: List[int] = [0, 1, 2, 3],
                 # 小目标层索引（默认0,1对应P2,P3）
                 small_levels: List[int] = [0, 1],
                 # 大目标层索引（默认2,3对应P4,P5）
                 large_levels: List[int] = [2, 3],
                 # 基础AFE配置（公共部分）
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
        # 为每个backbone level构建AFE模块（如果没有增强则Identity）
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
                # 注意：in_channels 应为 out_channels（因为作用在侧向连接之后）
                self.afe_modules.append(AngleFreqEnhance(in_channels=out_channels, **cfg))
            else:
                self.afe_modules.append(nn.Identity())

    @auto_fp16()
    def forward(self, inputs):
        # 构建侧向连接并应用AFE
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            feat = lateral_conv(inputs[i + self.start_level])
            feat = self.afe_modules[i](feat)
            laterals.append(feat)
        # Top-down 融合（与原FPN相同）
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i - 1] = laterals[i - 1] + upsampled
        # 输出层
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        # 额外层（P6等）
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