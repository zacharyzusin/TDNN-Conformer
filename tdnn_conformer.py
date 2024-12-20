import torch
from typing import Optional
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.tdnn import TDNNModule
from wenet.transformer.tdnn_encoder_layer import TDNNConformerEncoderLayer
from wenet.utils.class_utils import (
    WENET_ACTIVATION_CLASSES,
    WENET_MLP_CLASSES,
    WENET_ATTENTION_CLASSES,
)

class TDNNConformerEncoder(BaseEncoder):
    """TDNN-Conformer encoder module."""
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        tdnn_module_kernel: int = 3,
        tdnn_module_dilation: int = 1,
        tdnn_module_context_size: int = 2,
        causal: bool = False,
        use_tdnn_norm: str = "batch_norm",
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        conv_bias: bool = True,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        mlp_type: str = 'position_wise_feed_forward',
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
    ):
        """Construct TDNNConformerEncoder

        Args:
            input_size to use_dynamic_chunk (from BaseEncoder)
            tdnn_module_kernel (int): Kernel size for TDNN layers
            tdnn_module_dilation (int): Base dilation rate for TDNN
            tdnn_module_context_size (int): Number of TDNN layers/context size
            causal (bool): whether to use causal convolution or not
            use_tdnn_norm (str): normalization type for TDNN module
            activation_type (str): type of activation function.
            key_bias (bool): whether to use bias in attention.linear_k.
        """
        super().__init__(input_size, output_size, attention_heads,
                      linear_units, num_blocks, dropout_rate,
                      positional_dropout_rate, attention_dropout_rate,
                      input_layer, pos_enc_layer_type, normalize_before,
                      static_chunk_size, use_dynamic_chunk, global_cmvn,
                      use_dynamic_left_chunk, gradient_checkpointing,
                      use_sdpa, layer_norm_type, norm_eps)
        
        activation = WENET_ACTIVATION_CLASSES[activation_type]()

        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            query_bias,
            key_bias,
            value_bias,
            use_sdpa,
            n_kv_head,
            head_dim,
        )
        
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
            mlp_bias,
            n_expert,
            n_expert_activated,
        )

        mlp_class = WENET_MLP_CLASSES[mlp_type]

        self.encoders = torch.nn.ModuleList([
            TDNNConformerEncoderLayer(
                output_size,
                WENET_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                mlp_class(*positionwise_layer_args),
                mlp_class(*positionwise_layer_args),
                TDNNModule(
                    output_size,
                    kernel_size=tdnn_module_kernel,
                    dilation=tdnn_module_dilation,
                    context_size=tdnn_module_context_size,
                    activation=activation,
                    norm=use_tdnn_norm,
                    causal=causal,
                    bias=conv_bias,
                ),
                dropout_rate,
                normalize_before,
                layer_norm_type=layer_norm_type,
                norm_eps=norm_eps,
            ) for _ in range(num_blocks)
        ])
