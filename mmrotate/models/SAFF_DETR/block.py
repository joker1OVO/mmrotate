import torch
import torch.nn as nn

from mmrotate.models.SAFF_DETR.conv import Conv, RepConv
from mmrotate.models.builder import ROTATED_NECKS


__all__ = ["RepC3"]

@ROTATED_NECKS.register_module()
class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """
        Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))