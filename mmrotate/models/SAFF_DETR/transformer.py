from typing import List, Optional

import torch
import torch.nn as nn
from ..builder import ROTATED_NECKS

__all__ = ["AIFI"]

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    This class implements a standard transformer encoder layer with multi-head attention and feedforward network,
    supporting both pre-normalization and post-normalization configurations.

    Attributes:
        ma (nn.MultiheadAttention): Multi-head attention module.
        fc1 (nn.Linear): First linear layer in the feedforward network.
        fc2 (nn.Linear): Second linear layer in the feedforward network.
        norm1 (nn.LayerNorm): Layer normalization after attention.
        norm2 (nn.LayerNorm): Layer normalization after feedforward network.
        dropout (nn.Dropout): Dropout layer for the feedforward network.
        dropout1 (nn.Dropout): Dropout layer after attention.
        dropout2 (nn.Dropout): Dropout layer after feedforward network.
        act (nn.Module): Activation function.
        normalize_before (bool): Whether to apply normalization before attention and feedforward.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """
        Initialize the TransformerEncoderLayer with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__()
        """
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        """
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform forward pass with post-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform forward pass with pre-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward propagate the input through the encoder module.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after transformer encoder layer.
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

@ROTATED_NECKS.register_module()
class AIFI(TransformerEncoderLayer):
    """
    AIFI transformer layer for 2D data with positional embeddings.

    This class extends TransformerEncoderLayer to work with 2D feature maps by adding 2D sine-cosine positional
    embeddings and handling the spatial dimensions appropriately.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
    ):
        """
        Initialize the AIFI instance with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AIFI transformer layer.

        Args:
            x (torch.Tensor): Input tensor with shape [B, C, H, W].

        Returns:
            (torch.Tensor): Output tensor with shape [B, C, H, W].
        """
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0
    ) -> torch.Tensor:
        """
        Build 2D sine-cosine position embedding.

        Args:
            w (int): Width of the feature map.
            h (int): Height of the feature map.
            embed_dim (int): Embedding dimension.
            temperature (float): Temperature for the sine/cosine functions.

        Returns:
            (torch.Tensor): Position embedding with shape [1, embed_dim, h*w].
        """
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]