import torch
from torch import nn
from typing import Tuple, Optional
from wenet.utils.class_utils import WENET_NORM_CLASSES

class TDNNModule(nn.Module):
    """TDNN (Time Delay Neural Network) module for Conformer.
    
    This module replaces the ConvolutionModule in the original Wenet Conformer.
    It processes temporal information using dilated convolutions.
    
    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernel size of conv layers.
        dilation (int): Base dilation rate for the TDNN.
        context_size (int): Context frames to consider (-context_size, +context_size).
        activation (nn.Module): Activation function.
        norm (str): Normalization type.
        causal (bool): Whether use causal convolution or not.
        bias (bool): Whether to use bias in convolution layers.
        norm_eps (float): Epsilon value for normalization layers.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        context_size: int = 2,
        activation: nn.Module = nn.ReLU(),
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.tdnn_layers = nn.ModuleList()
        for i in range(context_size):
            current_dilation = dilation * (i + 1)
            if not causal:
                padding = (kernel_size - 1) * current_dilation // 2
            else:
                padding = 0
            self.tdnn_layers.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=current_dilation,
                    groups=channels,
                    bias=bias,
                )
            )

        self.lorder = max((kernel_size - 1) * dilation * context_size if causal else 0, 0)

        self.tdnn_combine = nn.Conv1d(
            channels * context_size,
            channels,
            kernel_size=1,
            bias=bias
        )

        assert norm in ['batch_norm', 'layer_norm', 'rms_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = WENET_NORM_CLASSES['batch_norm'](channels, eps=norm_eps)
        else:
            self.use_layer_norm = True
            self.norm = WENET_NORM_CLASSES[norm](channels, eps=norm_eps)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: Optional[torch.Tensor] = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute TDNN module.
        
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): Used for batch padding (#batch, 1, time).
            cache (torch.Tensor): Left context cache, it is only used in causal convolution.
                (#batch, channels, cache_t).
                
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
            torch.Tensor: New cache tensor.
        """
        x = x.transpose(1, 2) 

        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0) 
                assert cache.size(1) == x.size(1) 
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            new_cache = x[:, :, -self.lorder:]
        else:
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        x = self.pointwise_conv1(x) 
        x = nn.functional.glu(x, dim=1)

        tdnn_outputs = []
        for tdnn_layer in self.tdnn_layers:
            tdnn_outputs.append(tdnn_layer(x))
        
        x = torch.cat(tdnn_outputs, dim=1)
        x = self.tdnn_combine(x)

        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)

        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache