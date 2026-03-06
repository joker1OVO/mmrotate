import torch
import torch.nn as nn
from ..builder import ROTATED_NECKS


@ROTATED_NECKS.register_module()
class HighPassFilter(nn.Module):
    """
    对输入特征图进行高通滤波的模块（基于 FFT）。
    输入形状: (B, C, H, W)
    输出形状: 与输入相同
    """
    def __init__(self, alpha: float = 0.25, trainable_alpha: bool = False):
        """
        Args:
            alpha (float): 低频截止比例，取值 [0, 1]。0 表示全通，1 表示全阻。
            trainable_alpha (bool): 是否将 alpha 作为可训练参数。
        """
        super().__init__()
        if trainable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        # 实数 FFT (节省计算)
        x_fft = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W//2+1]

        # 生成频率坐标 (归一化到 [0,1])
        freq_h = torch.fft.fftfreq(H, device=x.device).view(-1, 1).abs()   # [H,1]
        freq_w = torch.fft.rfftfreq(W, device=x.device).view(1, -1).abs()  # [1, W//2+1]
        freq_h = freq_h / freq_h.max()
        freq_w = freq_w / freq_w.max()

        # 计算半径并生成高通掩码 (半径 >= alpha 的位置保留)
        radius = (freq_h ** 2 + freq_w ** 2) ** 0.5  # [H, W//2+1]
        mask = (radius >= self.alpha).float()        # [H, W//2+1]

        # 应用掩码
        x_fft_filtered = x_fft * mask.unsqueeze(0).unsqueeze(0)

        # 逆变换回空间域
        x_filtered = torch.fft.irfft2(x_fft_filtered, s=(H, W), norm='ortho')
        return x_filtered