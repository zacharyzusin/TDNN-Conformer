import torch
from torch import nn
from typing import Optional, Tuple
from wenet.transformer.attention import T_CACHE
from wenet.utils.class_utils import WENET_NORM_CLASSES

class TDNNConformerEncoderLayer(nn.Module):
    """Encoder layer module with TDNN instead of convolution.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
        feed_forward (torch.nn.Module): Feed-forward module instance.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
        tdnn_module (torch.nn.Module): TDNN module instance.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
        layer_norm_type (str): type of layer norm
        norm_eps (float): epsilon value for layer norm
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        tdnn_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.tdnn_module = tdnn_module
        self.norm_ff = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
        self.norm_mha = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)

        if feed_forward_macaron is not None:
            self.norm_ff_macaron = WENET_NORM_CLASSES[layer_norm_type](
                size, eps=norm_eps)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0

        if self.tdnn_module is not None:
            self.norm_tdnn = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)
            self.norm_final = WENET_NORM_CLASSES[layer_norm_type](size, eps=norm_eps)

        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.size = size

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: Optional[T_CACHE] = None,
        tdnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time, time).
            pos_emb (torch.Tensor): Positional embedding tensor.
            mask_pad (torch.Tensor): Padding mask tensor (#batch, 1, time).
            att_cache (torch.Tensor): Cache tensor of attention.
            tdnn_cache (torch.Tensor): Cache tensor of TDNN.

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor.
            torch.Tensor: tdnn_cache tensor.
        """
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if att_cache is None:
            att_cache = (torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0, 0, 0)))

        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, pos_emb, cache=att_cache)

        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        if self.tdnn_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_tdnn(x)
            if tdnn_cache is None:
                tdnn_cache = torch.zeros((0, 0, 0), device=x.device)
            x, new_tdnn_cache = self.tdnn_module(x, mask_pad, tdnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_tdnn(x)
        else:
            new_tdnn_cache = torch.zeros((0, 0, 0), device=x.device)

        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.tdnn_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_tdnn_cache