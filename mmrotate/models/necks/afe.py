import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List

EPS = 1e-8

def estimate_main_direction_batch(patch_tensor, eps=EPS):
    """
    patch_tensor: [B*N, 1, m, m]
    Returns: [B*N] angles in [0, pi)
    """
    Bn, _, m, _ = patch_tensor.shape
    device = patch_tensor.device
    x_fft = torch.fft.fft2(patch_tensor.squeeze(1), norm='ortho')
    x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
    mag = x_fft_shifted.abs() + eps

    h_freq = torch.fft.fftfreq(m, d=1.0) * m
    w_freq = torch.fft.fftfreq(m, d=1.0) * m
    h_grid, w_grid = torch.meshgrid(h_freq, w_freq, indexing='ij')
    h_grid = torch.fft.fftshift(h_grid).to(device)
    w_grid = torch.fft.fftshift(w_grid).to(device)

    rho = torch.sqrt(h_grid ** 2 + w_grid ** 2)
    theta = torch.atan2(h_grid, w_grid)
    theta = (theta + 2 * math.pi) % (2 * math.pi)

    mask = rho > eps
    rho_valid = rho[mask]
    theta_valid = theta[mask]

    mag_flat = mag.view(Bn, -1)
    mag_valid = mag_flat[:, mask.view(-1)]
    weighted_energy = mag_valid * rho_valid.unsqueeze(0)
    max_idx = torch.argmax(weighted_energy, dim=1)
    theta_e = theta_valid[max_idx]
    return theta_e % math.pi

def compute_angle_map(x, window_size=7, stride=1):
    B, C, H, W = x.shape
    x_mean = x.mean(dim=1, keepdim=True)
    pad = window_size // 2
    x_pad = F.pad(x_mean, (pad, pad, pad, pad), mode='reflect')
    patches = F.unfold(x_pad, kernel_size=window_size, stride=stride)
    N = patches.shape[-1]
    patches = patches.transpose(1, 2).reshape(B * N, 1, window_size, window_size)
    angles = estimate_main_direction_batch(patches)
    angle_map = angles.view(B, H, W)
    angle_map = torch.nan_to_num(angle_map, nan=0.0, posinf=math.pi, neginf=0.0)
    return angle_map

class DynamicDirectionalConv(nn.Module):
    def __init__(self, in_channels, mid_channels=16, kernel_size=7, stride=1, padding=None):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

        self.reduce = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.expand = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.angle_bias = nn.Parameter(torch.tensor(0.0))

        # 可学习的基础卷积核（深度卷积形式）
        # 初始化：高斯平滑核，有助于稳定性
        self.base_kernel = nn.Parameter(
            self._create_gaussian_kernel(kernel_size, sigma=0.5).repeat(mid_channels, 1, 1, 1)
        )

    def _create_gaussian_kernel(self, k, sigma):
        ax = torch.linspace(-(k//2), k//2, k)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2*sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # [1,1,k,k]

    def _rotate_kernel(self, kernel, angle):
        """
        kernel: [mid, 1, k, k]
        angle: scalar Tensor
        Returns: rotated kernel of same shape
        """
        mid, _, k, _ = kernel.shape
        device = kernel.device
        center = (k - 1) / 2.0
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        # 仿射矩阵（逆时针旋转）
        theta = torch.tensor([
            [cos_a, -sin_a, center - cos_a*center + sin_a*center],
            [sin_a,  cos_a, center - sin_a*center - cos_a*center]
        ], device=device).unsqueeze(0)  # [1,2,3]
        # kernel 形状 [mid, 1, k, k] -> 扩展 batch 维 [mid, 1, k, k] 需要 unsqueeze(0) 变成 [1, mid, 1, k, k]
        # 但 grid_sample 要求输入 [N, C, H, W]，这里 N=mid, C=1, H=W=k
        # 所以将 mid 视为 batch 维度
        kernel_4d = kernel.view(mid, 1, k, k)  # [mid, 1, k, k]
        grid = F.affine_grid(theta.expand(mid, -1, -1), kernel_4d.size(), align_corners=False)
        rotated = F.grid_sample(kernel_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return rotated  # [mid, 1, k, k]

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        k = self.kernel_size
        pad = self.padding
        mid = self.mid_channels

        # 1. 降维
        x_low = self.reduce(x)  # [B, mid, H, W]

        # 2. 计算角度图（借用原有方法，它是在低维特征上计算）
        angle_fft = compute_angle_map(x_low, window_size=k, stride=self.stride)
        angle = angle_fft + self.angle_bias
        angle = angle % math.pi
        # 限制角度范围，避免过大偏移
        angle = torch.clamp(angle, 0, math.pi - 1e-6)

        # 3. 获取 patches
        x_pad = F.pad(x_low, (pad, pad, pad, pad), mode='reflect')
        patches = F.unfold(x_pad, kernel_size=k, stride=self.stride)  # [B, mid*k*k, N]
        N = patches.shape[-1]  # H*W
        patches = patches.view(B, mid, k*k, N).permute(0, 1, 3, 2)  # [B, mid, N, k*k]

        # 4. 对每个位置，旋转基础核，然后与 patch 做点积
        # 展平空间位置
        angle_flat = angle.reshape(-1)  # [B*N]
        # 输出容器
        out_low_flat = torch.zeros(B*N, mid, device=device)

        # 遍历每个位置（可优化，但先保证正确性）
        for idx in range(B*N):
            a = angle_flat[idx]
            # 旋转基础核
            rot_kernel = self._rotate_kernel(self.base_kernel, a)  # [mid, 1, k, k]
            rot_kernel_flat = rot_kernel.view(mid, k*k)  # [mid, k*k]
            # 取对应 patch
            patches_flat = patches.reshape(B * N, mid, k * k)  # 先 reshape 成 [B*N, mid, k*k]
            patch = patches_flat[idx]  # 取第 idx 个
            # 逐通道点积
            out_val = (patch * rot_kernel_flat).sum(dim=1)  # [mid]
            out_low_flat[idx] = out_val

        out_low = out_low_flat.view(B, mid, H, W)
        out = self.expand(out_low)
        return out


@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    """
    增强版 FPN，支持可配置的 AFE 模块参数
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes: List[str],
                 afe_mid_channels=16,      # 新增：降维通道数
                 afe_kernel_size=7,        # 新增：局部窗口大小 / 卷积核大小
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs)

        self.fusion_modes = fusion_modes
        self.dynamic_convs = nn.ModuleList()
        for mode in fusion_modes:
            if mode == 'afe':
                self.dynamic_convs.append(
                    DynamicDirectionalConv(
                        out_channels,
                        mid_channels=afe_mid_channels,
                        kernel_size=afe_kernel_size
                    )
                )
            else:
                self.dynamic_convs.append(None)

    @auto_fp16()
    def forward(self, inputs):
        # 同原始 FPN 的 lateral 构建
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        # 自顶向下融合
        for i in range(used_backbone_levels - 1, 0, -1):
            fusion_idx = used_backbone_levels - 1 - i
            mode = self.fusion_modes[fusion_idx]

            if mode == 'add':
                if 'scale_factor' in self.upsample_cfg:
                    upsampled = F.interpolate(laterals[i], **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
                laterals[i - 1] = laterals[i - 1] + upsampled

            elif mode == 'afe':
                enhanced_low = self.dynamic_convs[fusion_idx](laterals[i - 1])
                up_high = F.interpolate(laterals[i], size=enhanced_low.shape[-2:], **self.upsample_cfg)
                laterals[i - 1] = enhanced_low + up_high

            else:
                raise ValueError(f"Unknown fusion mode: {mode}")

        # 输出层
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # 生成额外层（同原 FPN）
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