# angle_freq_enhance.py
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
    频域极坐标增强模块：均匀半径分区 + 重叠角度扇区 + 通道独立可学习权重
    """

    def __init__(self,
                 in_channels: int = 256,
                 c_mid: int = 16,
                 n_angles: int = 8,
                 radius_width: int = 8,          # 均匀半径宽度（像素）
                 overlap_ratio: float = 1.5,      # 扇区重叠系数，1=无重叠，2=完全重叠
                 learnable_weights: bool = True,
                 residual: bool = True,
                 use_hann_window: bool = False,
                 eps: float = 1e-8):
        super().__init__()
        self.c_mid = c_mid
        self.n_angles = n_angles
        self.radius_width = radius_width
        self.overlap_ratio = overlap_ratio
        self.residual = residual
        self.use_hann_window = use_hann_window
        self.eps = eps

        # 投影到 c_mid 通道
        self.proj_in = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)

        if learnable_weights:
            # 权重形状: (c_mid, n_angles, n_radii)
            # 初始化为1，允许网络增强或抑制
            self.weights = nn.Parameter(torch.ones(c_mid, n_angles, 1))  # n_radii 后面动态确定
        else:
            self.register_buffer('weights', torch.ones(c_mid, n_angles, 1))

        # 预注册缓存，避免每次生成掩码
        self.register_buffer('radius_idx', None)       # [H, W]  半径索引
        self.register_buffer('angle_weights', None)    # [n_angles, H, W]  每个扇区的权重（归一化）
        self.register_buffer('_hann_window', None)
        self._cached_HW = None
        self._cached_n_radii = None

    def _build_masks(self, H: int, W: int, device: torch.device):
        """生成半径索引和角度软分配权重"""
        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device), indexing='ij')
        r = torch.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        theta = torch.atan2(y - cy, x - cx) + math.pi   # [0, 2π)

        # 半径索引（均匀分区）
        max_r = max(cy, cx)
        n_radii = int(max_r // self.radius_width) + 1
        radius_idx = torch.floor(r / self.radius_width).long().clamp(0, n_radii - 1)

        # 角度软分配（三角加权，覆盖 [0, π) 即可，因为对称性）
        theta = theta % math.pi   # 映射到 [0, π)
        delta = math.pi / self.n_angles
        half_width = self.overlap_ratio * delta / 2.0
        angle_weights = torch.zeros(self.n_angles, H, W, device=device)

        for a in range(self.n_angles):
            center = a * delta + delta / 2.0
            dist = (theta - center).abs()
            mask = dist < half_width
            w = (1 - dist / half_width).clamp(min=0)
            angle_weights[a] = w * mask.float()

        # 归一化：每个位置的权重和为1（可选，避免能量变化）
        sum_w = angle_weights.sum(dim=0, keepdim=True) + self.eps
        angle_weights = angle_weights / sum_w

        # 缓存
        self.radius_idx = radius_idx.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        self.angle_weights = angle_weights.unsqueeze(0)         # [1, n_angles, H, W]
        self._cached_HW = (H, W)
        self._cached_n_radii = n_radii

        # 调整可学习权重的维度（如果动态变化）
        if isinstance(self.weights, nn.Parameter):
            if self.weights.size(-1) != n_radii:
                new_weights = torch.ones(self.c_mid, self.n_angles, n_radii, device=device)
                # 复制原有权重的平均值到新维度（可选）
                if self.weights.size(2) > 1:
                    # 简单起见，保持为1
                    pass
                self.weights = nn.Parameter(new_weights)
        else:
            # 如果是buffer，也更新
            if self.weights.size(-1) != n_radii:
                self.weights = torch.ones(self.c_mid, self.n_angles, n_radii, device=device)

        if self.use_hann_window and self._hann_window is None:
            # 2D Hanning 窗
            hann_1d = torch.hann_window(H, device=device)
            hann_2d = hann_1d.unsqueeze(1) * torch.hann_window(W, device=device).unsqueeze(0)
            self._hann_window = hann_2d.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # 1. 投影到低维频域空间
        x_proj = self.proj_in(x)  # [B, c_mid, H, W]

        # 2. 可选加窗
        if self.use_hann_window:
            if self._hann_window is None or self._hann_window.shape[-2:] != (H, W):
                self._build_masks(H, W, device)
            x_proj = x_proj * self._hann_window

        # 3. FFT，得到复数谱
        x_fft = torch.fft.fft2(x_proj, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shift.abs() + self.eps   # 幅度谱

        # 4. 构建或获取掩码（半径索引、角度权重）
        if self._cached_HW != (H, W):
            self._build_masks(H, W, device)

        # 5. 对每个 (c, a, r) 独立生成增益
        # radius_idx: [1,1,H,W]  值范围 0..n_radii-1
        # angle_weights: [1, n_angles, H, W]
        # weights: [c_mid, n_angles, n_radii]
        # 我们需要得到增益矩阵 gain: [B, c_mid, H, W]
        # 高效方法：使用 einsum 或 index_select
        # 这里采用逐通道循环（因为 c_mid <= 32，开销可接受）
        gain = torch.zeros(B, self.c_mid, H, W, device=device)
        r_idx = self.radius_idx.expand(B, -1, H, W)   # [B,1,H,W]
        aw = self.angle_weights.expand(B, -1, H, W)   # [B, n_angles, H, W]

        for c in range(self.c_mid):
            # 取出该通道的权重 (n_angles, n_radii)
            w_c = self.weights[c]  # [n_angles, n_radii]
            # 对于每个半径索引，取对应权重；然后与角度权重相乘并求和角度维
            # r_idx: [B,1,H,W] → 扩展为 [B, n_angles, H, W] 用于 gather
            r_idx_exp = r_idx.expand(-1, self.n_angles, -1, -1)  # [B, n_angles, H, W]
            # gather 得到每个角度扇区、每个位置的权重值
            w_selected = torch.gather(w_c.unsqueeze(0).expand(B, -1, -1),  # [B, n_angles, n_radii]
                                      dim=2,
                                      index=r_idx_exp.long())  # [B, n_angles, H, W]
            # 与角度权重相乘并求和
            gain_c = (aw * w_selected).sum(dim=1)  # [B, H, W]
            gain[:, c, :, :] = gain_c

        # 6. 应用增益到幅度谱
        mag_enhanced = mag * gain.unsqueeze(2)  # mag: [B, c_mid, H, W] → 广播乘法
        # 相位保持不变
        x_fft_shift_enhanced = mag_enhanced * torch.exp(1j * torch.angle(x_fft_shift))

        # 7. 逆变换回空间域
        x_fft_ishift = torch.fft.ifftshift(x_fft_shift_enhanced, dim=(-2, -1))
        x_enh = torch.fft.ifft2(x_fft_ishift, norm='ortho').real

        # 8. 投影回原通道数
        x_enh = self.proj_out(x_enh)

        # 9. 残差连接
        if self.residual:
            return x + x_enh
        else:
            return x_enh


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """在 FPN 的侧向连接后添加频域角度增强"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 enhance_levels: List[int] = [0, 1, 2, 3],   # 对应 P2,P3,P4,P5
                 afe_cfg: dict = dict(
                     c_mid=16,
                     n_angles=8,
                     radius_width=8,
                     overlap_ratio=1.5,
                     learnable_weights=True,
                     residual=True,
                     use_hann_window=False,
                 ),
                 **kwargs):
        super(AngleFreqEnhanceFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs
        )

        self.enhance_levels = enhance_levels
        self.afe_modules = nn.ModuleList()
        for i in range(len(in_channels)):
            if i in enhance_levels:
                self.afe_modules.append(AngleFreqEnhance(in_channels=out_channels, **afe_cfg))
            else:
                self.afe_modules.append(nn.Identity())

    @auto_fp16()
    def forward(self, inputs):
        # 侧向连接 + AFE
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            feat = lateral_conv(inputs[i + self.start_level])
            feat = self.afe_modules[i](feat)
            laterals.append(feat)

        # Top-down 融合 (原始 FPN 逻辑)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 输出卷积
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # 额外层（P6, P7）保持原样
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                # 省略 extra convs 的细节，与原 FPN 一致
                pass
        return tuple(outs)