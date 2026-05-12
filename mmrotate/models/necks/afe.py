import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.necks.fpn import FPN
from ..builder import ROTATED_NECKS
from typing import List

EPS = 1e-8

# ========== 角度估计函数（与之前相同，略作简化）==========
def estimate_main_direction_batch(patch_tensor, eps=EPS):
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


# ========== 旋转自适应卷积（深度可分离，角度窗口=卷积核尺寸）==========
class AngleAdaptiveConv(nn.Module):
    def __init__(self, in_channels, kernel_size=7, stride=1, padding=None, num_angles=60):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding
        self.num_angles = num_angles

        self.base_kernel = nn.Parameter(
            self._gaussian_kernel(kernel_size, sigma=0.5).repeat(in_channels, 1, 1, 1)
        )
        self.angle_bias = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('angle_grid', torch.linspace(0, math.pi, num_angles, dtype=torch.float32))

    def _gaussian_kernel(self, k, sigma):
        ax = torch.linspace(-(k//2), k//2, k)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2*sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def _rotate_kernel(self, kernel, angle):
        C, _, k, _ = kernel.shape
        device = kernel.device
        center = (k - 1) / 2.0
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        theta = torch.tensor([
            [cos_a, -sin_a, center - cos_a*center + sin_a*center],
            [sin_a,  cos_a, center - sin_a*center - cos_a*center]
        ], device=device).unsqueeze(0)
        kernel_4d = kernel.view(C, 1, k, k)
        grid = F.affine_grid(theta.expand(C, -1, -1), kernel_4d.size(), align_corners=False)
        rotated = F.grid_sample(kernel_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return rotated

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.kernel_size
        pad = self.padding
        # 角度估计（不加 90°）
        angle_fft = compute_angle_map(x, window_size=k, stride=self.stride)
        angle = angle_fft + self.angle_bias
        angle = angle % math.pi
        angle = torch.clamp(angle, 0, math.pi - 1e-6)

        # 预旋转核
        pre_rotated = []
        for i in range(self.num_angles):
            theta_i = self.angle_grid[i]
            rot_k = self._rotate_kernel(self.base_kernel, theta_i)
            pre_rotated.append(rot_k.view(C, k*k))
        pre_rotated = torch.stack(pre_rotated, dim=0)  # [num_angles, C, k*k]

        # 提取 patches
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        patches = F.unfold(x_pad, kernel_size=k, stride=self.stride)  # [B, C*k*k, N]
        N = patches.shape[-1]
        patches = patches.view(B, C, k*k, N).permute(0, 1, 3, 2)      # [B, C, N, k*k]
        patches_flat = patches.permute(0, 2, 1, 3).reshape(B*N, C, k*k) # [B*N, C, k*k]

        # 量化角度
        angle_flat = angle.reshape(-1)
        indices = torch.argmin(torch.abs(angle_flat.unsqueeze(1) - self.angle_grid.unsqueeze(0)), dim=1)
        selected_kernels = pre_rotated[indices]   # [B*N, C, k*k]
        out_flat = (patches_flat * selected_kernels).sum(dim=2)
        out = out_flat.view(B, C, H, W)
        return out


# ========== 空间注意力（CBAM 风格）==========
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(concat))
        return x * att


# ========== 通道注意力（SE）==========
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ========== 多尺度旋转分支（三个子分支，通道注意力加权融合）==========
class MultiScaleRotBranch(nn.Module):
    def __init__(self, in_channels=16, kernel_list=[5,7,9], reduction=4):
        super().__init__()
        self.branch_channels = in_channels   # 保存输入通道数
        self.num_branches = len(kernel_list)
        self.rot_convs = nn.ModuleList()
        self.spatial_attns = nn.ModuleList()
        for k in kernel_list:
            self.rot_convs.append(AngleAdaptiveConv(in_channels, kernel_size=k))
            self.spatial_attns.append(SpatialAttention(kernel_size=7))
        # 通道注意力用于融合三个分支的输出（拼接后 channels = in_channels * num_branches）
        self.fusion_att = ChannelAttention(in_channels * self.num_branches, reduction=reduction)

    def forward(self, x):
        branch_outs = []
        for rot, attn in zip(self.rot_convs, self.spatial_attns):
            out = rot(x)
            out = attn(out)
            branch_outs.append(out)   # 每个 [B, self.branch_channels, H, W]
        # 拼接
        concat = torch.cat(branch_outs, dim=1)   # [B, self.branch_channels * num_branches, H, W]
        # 通道注意力生成权重
        weight_map = self.fusion_att(concat)     # [B, total_channels, H, W]
        # 拆分成三组，分别加权原分支输出
        weighted_outs = []
        start = 0
        for i in range(self.num_branches):
            # 使用 self.branch_channels 而不是未定义的 in_channels
            w = weight_map[:, start:start+self.branch_channels, :, :]
            weighted_outs.append(branch_outs[i] * w)
            start += self.branch_channels
        # 逐元素相加
        out = torch.stack(weighted_outs, dim=0).sum(dim=0)   # [B, self.branch_channels, H, W]
        return out


# ========== 最终模块 ==========
class MultiBranchAFE(nn.Module):
    def __init__(self, in_channels=256, reduced_channels=16, kernel_list=[5,7,9]):
        super().__init__()
        self.in_channels = in_channels
        self.reduced_channels = reduced_channels

        # 分支 A: 3x3 标准卷积（无偏置）
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn3x3 = nn.GroupNorm(16, in_channels)   # 256 channels -> 16 groups
        self.act = nn.SiLU()

        # 分支 B: 降维 + 多尺度旋转分支 + 升维
        self.reduce = nn.Conv2d(in_channels, reduced_channels, 1, bias=False)
        self.gn_reduce = nn.GroupNorm(4, reduced_channels)  # 16 channels -> 4 groups
        self.multi_rot = MultiScaleRotBranch(reduced_channels, kernel_list)
        self.expand = nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        self.gn_expand = nn.GroupNorm(16, in_channels)

        # 全局组归一化和激活（最后输出前）
        self.gn_out = nn.GroupNorm(16, in_channels)

    def forward(self, x):
        # 分支 A
        out_a = self.conv3x3(x)
        out_a = self.gn3x3(out_a)
        out_a = self.act(out_a)

        # 分支 B
        x_low = self.reduce(x)
        x_low = self.gn_reduce(x_low)
        x_low = self.act(x_low)
        out_b_low = self.multi_rot(x_low)            # [B,16,H,W]
        out_b = self.expand(out_b_low)
        out_b = self.gn_expand(out_b)
        out_b = self.act(out_b)

        # 融合（逐元素平均）
        out = (out_a + out_b) / 2.0

        # 全局残差 + 最终归一化 + 激活
        out = out + x
        out = self.gn_out(out)
        out = self.act(out)
        return out


# ========== FPN 包装 ==========
@ROTATED_NECKS.register_module()
class AngleFreqEnhanceFPN(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes: List[str],
                 afe_reduced_channels=16,
                 afe_kernel_list=[5,7,9],
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
                    MultiBranchAFE(
                        in_channels=out_channels,
                        reduced_channels=afe_reduced_channels,
                        kernel_list=afe_kernel_list
                    )
                )
            else:
                self.dynamic_convs.append(None)

    @auto_fp16()
    def forward(self, inputs):
        # 标准 FPN lateral 构建
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
        # 输出层和额外层（与原 FPN 相同）
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
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)