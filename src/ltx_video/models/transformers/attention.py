import inspect
from importlib import import_module
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention import _chunked_feed_forward
from diffusers.models.attention_processor import (
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    SpatialNorm,
)
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.models.normalization import RMSNorm
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import rearrange
from torch import nn

from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

try:
    from torch_xla.experimental.custom_kernel import flash_attention
except ImportError:
    # workaround for automatic tests. Currently this function is manually patched
    # to the torch_xla lib on setup of container
    pass

# code adapted from  https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

logger = logging.get_logger(__name__)


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        qk_norm (`str`, *optional*, defaults to None):
            Set to 'layer_norm' or `rms_norm` to perform query and key normalization.
        adaptive_norm (`str`, *optional*, defaults to `"single_scale_shift"`):
            The type of adaptive norm to use. Can be `"single_scale_shift"`, `"single_scale"` or "none".
        standardization_norm (`str`, *optional*, defaults to `"layer_norm"`):
            The type of pre-normalization to use. Can be `"layer_norm"` or `"rms_norm"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,  # pylint: disable=unused-argument
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        adaptive_norm: str = "single_scale_shift",  # 'single_scale_shift', 'single_scale' or 'none'
        standardization_norm: str = "layer_norm",  # 'layer_norm' or 'rms_norm'
        norm_eps: float = 1e-5,
        qk_norm: Optional[str] = None,
        final_dropout: bool = False,
        attention_type: str = "default",  # pylint: disable=unused-argument
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        use_tpu_flash_attention: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_tpu_flash_attention = use_tpu_flash_attention
        self.adaptive_norm = adaptive_norm

        assert standardization_norm in ["layer_norm", "rms_norm"]
        assert adaptive_norm in ["single_scale_shift", "single_scale", "none"]

        make_norm_layer = (
            nn.LayerNorm if standardization_norm == "layer_norm" else RMSNorm
        )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = make_norm_layer(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
        )

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            use_tpu_flash_attention=use_tpu_flash_attention,
            qk_norm=qk_norm,
            use_rope=use_rope,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=(
                    cross_attention_dim if not double_self_attention else None
                ),
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                use_tpu_flash_attention=use_tpu_flash_attention,
                qk_norm=qk_norm,
                use_rope=use_rope,
            )  # is self-attn if encoder_hidden_states is none

            if adaptive_norm == "none":
                self.attn2_norm = make_norm_layer(
                    dim, norm_eps, norm_elementwise_affine
                )
        else:
            self.attn2 = None
            self.attn2_norm = None

        self.norm2 = make_norm_layer(dim, norm_eps, norm_elementwise_affine)

        # 3. Feed-forward
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 5. Scale-shift for PixArt-Alpha.
        if adaptive_norm != "none":
            num_ada_params = 4 if adaptive_norm == "single_scale" else 6
            self.scale_shift_table = nn.Parameter(
                torch.randn(num_ada_params, dim) / dim**0.5
            )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_use_tpu_flash_attention(self):
        r"""
        Function sets the flag in this object and propagates down the children. The flag will enforce the usage of TPU
        attention kernel.
        """
        self.use_tpu_flash_attention = True
        self.attn1.set_use_tpu_flash_attention()
        self.attn2.set_use_tpu_flash_attention()

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        freqs_cis: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        skip_layer_mask: Optional[torch.Tensor] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
    ) -> torch.FloatTensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored."
                )

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        original_hidden_states = hidden_states

        norm_hidden_states = self.norm1(hidden_states)

        # Apply ada_norm_single
        if self.adaptive_norm in ["single_scale_shift", "single_scale"]:
            assert timestep.ndim == 3  # [batch, 1 or num_tokens, embedding_dim]
            num_ada_params = self.scale_shift_table.shape[0]
            ada_values = self.scale_shift_table[None, None] + timestep.reshape(
                batch_size, timestep.shape[1], num_ada_params, -1
            )
            if self.adaptive_norm == "single_scale_shift":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    ada_values.unbind(dim=2)
                )
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            else:
                scale_msa, gate_msa, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa)
        elif self.adaptive_norm == "none":
            scale_msa, gate_msa, scale_mlp, gate_mlp = None, None, None, None
        else:
            raise ValueError(f"Unknown adaptive norm type: {self.adaptive_norm}")

        norm_hidden_states = norm_hidden_states.squeeze(
            1
        )  # TODO: Check if this is needed

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )

        attn_output = self.attn1(
            norm_hidden_states,
            freqs_cis=freqs_cis,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            skip_layer_mask=skip_layer_mask,
            skip_layer_strategy=skip_layer_strategy,
            **cross_attention_kwargs,
        )
        if gate_msa is not None:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.adaptive_norm == "none":
                attn_input = self.attn2_norm(hidden_states)
            else:
                attn_input = hidden_states
            attn_output = self.attn2(
                attn_input,
                freqs_cis=freqs_cis,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        if self.adaptive_norm == "single_scale_shift":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        elif self.adaptive_norm == "single_scale":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp)
        elif self.adaptive_norm == "none":
            pass
        else:
            raise ValueError(f"Unknown adaptive norm type: {self.adaptive_norm}")

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            ff_output = self.ff(norm_hidden_states)
        if gate_mlp is not None:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        if (
            skip_layer_mask is not None
            and skip_layer_strategy == SkipLayerStrategy.TransformerBlock
        ):
            skip_layer_mask = skip_layer_mask.view(-1, 1, 1)
            hidden_states = hidden_states * skip_layer_mask + original_hidden_states * (
                1.0 - skip_layer_mask
            )

        return hidden_states


@maybe_allow_in_graph
class Attention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        qk_norm (`str`, *optional*, defaults to None):
            Set to 'layer_norm' or `rms_norm` to perform query and key normalization.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        qk_norm: Optional[str] = None,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        use_tpu_flash_attention: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.use_tpu_flash_attention = use_tpu_flash_attention
        self.use_rope = use_rope

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        if qk_norm is None:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head * heads, eps=1e-5)
            self.k_norm = RMSNorm(dim_head * heads, eps=1e-5)
        elif qk_norm == "layer_norm":
            self.q_norm = nn.LayerNorm(dim_head * heads, eps=1e-5)
            self.k_norm = nn.LayerNorm(dim_head * heads, eps=1e-5)
        else:
            raise ValueError(f"Unsupported qk_norm method: {qk_norm}")

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(
                num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True
            )
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(
                f_channels=query_dim, zq_channels=spatial_norm_dim
            )
        else:
            self.spatial_norm = None

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels,
                num_groups=cross_attention_norm_num_groups,
                eps=1e-5,
                affine=True,
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        linear_cls = nn.Linear

        self.linear_cls = linear_cls
        self.to_q = linear_cls(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = linear_cls(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_v = linear_cls(self.cross_attention_dim, self.inner_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = linear_cls(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = linear_cls(added_kv_proj_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(linear_cls(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:
            processor = AttnProcessor2_0()
        self.set_processor(processor)

    def set_use_tpu_flash_attention(self):
        r"""
        Function sets the flag in this object. The flag will enforce the usage of TPU attention kernel.
        """
        self.use_tpu_flash_attention = True

    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(
                f"You are removing possibly trained weights of {self.processor} with {processor}"
            )
            self._modules.pop("processor")

        self.processor = processor

    def get_processor(
        self, return_deprecated_lora: bool = False
    ) -> "AttentionProcessor":  # noqa: F821
        r"""
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        if not return_deprecated_lora:
            return self.processor

        # TODO(Sayak, Patrick). The rest of the function is needed to ensure backwards compatible
        # serialization format for LoRA Attention Processors. It should be deleted once the integration
        # with PEFT is completed.
        is_lora_activated = {
            name: module.lora_layer is not None
            for name, module in self.named_modules()
            if hasattr(module, "lora_layer")
        }

        # 1. if no layer has a LoRA activated we can return the processor as usual
        if not any(is_lora_activated.values()):
            return self.processor

        # If doesn't apply LoRA do `add_k_proj` or `add_v_proj`
        is_lora_activated.pop("add_k_proj", None)
        is_lora_activated.pop("add_v_proj", None)
        # 2. else it is not posssible that only some layers have LoRA activated
        if not all(is_lora_activated.values()):
            raise ValueError(
                f"Make sure that either all layers or no layers have LoRA activated, but have {is_lora_activated}"
            )

        # 3. And we need to merge the current LoRA layers into the corresponding LoRA attention processor
        non_lora_processor_cls_name = self.processor.__class__.__name__
        lora_processor_cls = getattr(
            import_module(__name__), "LoRA" + non_lora_processor_cls_name
        )

        hidden_size = self.inner_dim

        # now create a LoRA attention processor from the LoRA layers
        if lora_processor_cls in [
            LoRAAttnProcessor,
            LoRAAttnProcessor2_0,
            LoRAXFormersAttnProcessor,
        ]:
            kwargs = {
                "cross_attention_dim": self.cross_attention_dim,
                "rank": self.to_q.lora_layer.rank,
                "network_alpha": self.to_q.lora_layer.network_alpha,
                "q_rank": self.to_q.lora_layer.rank,
                "q_hidden_size": self.to_q.lora_layer.out_features,
                "k_rank": self.to_k.lora_layer.rank,
                "k_hidden_size": self.to_k.lora_layer.out_features,
                "v_rank": self.to_v.lora_layer.rank,
                "v_hidden_size": self.to_v.lora_layer.out_features,
                "out_rank": self.to_out[0].lora_layer.rank,
                "out_hidden_size": self.to_out[0].lora_layer.out_features,
            }

            if hasattr(self.processor, "attention_op"):
                kwargs["attention_op"] = self.processor.attention_op

            lora_processor = lora_processor_cls(hidden_size, **kwargs)
            lora_processor.to_q_lora.load_state_dict(self.to_q.lora_layer.state_dict())
            lora_processor.to_k_lora.load_state_dict(self.to_k.lora_layer.state_dict())
            lora_processor.to_v_lora.load_state_dict(self.to_v.lora_layer.state_dict())
            lora_processor.to_out_lora.load_state_dict(
                self.to_out[0].lora_layer.state_dict()
            )
        elif lora_processor_cls == LoRAAttnAddedKVProcessor:
            lora_processor = lora_processor_cls(
                hidden_size,
                cross_attention_dim=self.add_k_proj.weight.shape[0],
                rank=self.to_q.lora_layer.rank,
                network_alpha=self.to_q.lora_layer.network_alpha,
            )
            lora_processor.to_q_lora.load_state_dict(self.to_q.lora_layer.state_dict())
            lora_processor.to_k_lora.load_state_dict(self.to_k.lora_layer.state_dict())
            lora_processor.to_v_lora.load_state_dict(self.to_v.lora_layer.state_dict())
            lora_processor.to_out_lora.load_state_dict(
                self.to_out[0].lora_layer.state_dict()
            )

            # only save if used
            if self.add_k_proj.lora_layer is not None:
                lora_processor.add_k_proj_lora.load_state_dict(
                    self.add_k_proj.lora_layer.state_dict()
                )
                lora_processor.add_v_proj_lora.load_state_dict(
                    self.add_v_proj.lora_layer.state_dict()
                )
            else:
                lora_processor.add_k_proj_lora = None
                lora_processor.add_v_proj_lora = None
        else:
            raise ValueError(f"{lora_processor_cls} does not exist.")

        return lora_processor

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        freqs_cis: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        skip_layer_mask: Optional[torch.Tensor] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            skip_layer_mask (`torch.Tensor`, *optional*):
                The skip layer mask to use. If `None`, no mask is applied.
            skip_layer_strategy (`SkipLayerStrategy`, *optional*, defaults to `None`):
                Controls which layers to skip for spatiotemporal guidance.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by"
                f" {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {
            k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters
        }

        return self.processor(
            self,
            hidden_states,
            freqs_cis=freqs_cis,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            skip_layer_mask=skip_layer_mask,
            skip_layer_strategy=skip_layer_strategy,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size // head_size, seq_len, dim * head_size
        )
        return tensor

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """

        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(
            batch_size, seq_len * extra_dim, head_size, dim // head_size
        )
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(
                batch_size * head_size, seq_len * extra_dim, dim // head_size
            )

        return tensor

    def get_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        target_length: int,
        batch_size: int,
        out_dim: int = 3,
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    target_length,
                )
                padding = torch.zeros(
                    padding_shape,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def norm_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`torch.Tensor`): Hidden states of the encoder.

        Returns:
            `torch.Tensor`: The normalized encoder hidden states.
        """
        assert (
            self.norm_cross is not None
        ), "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states

    @staticmethod
    def apply_rotary_emb(
        input_tensor: torch.Tensor,
        freqs_cis: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_freqs = freqs_cis[0]
        sin_freqs = freqs_cis[1]

        t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
        t1, t2 = t_dup.unbind(dim=-1)
        t_dup = torch.stack((-t2, t1), dim=-1)
        input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

        out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs

        return out


class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        freqs_cis: Tuple[torch.FloatTensor, torch.FloatTensor],
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        skip_layer_mask: Optional[torch.FloatTensor] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if skip_layer_mask is not None:
            skip_layer_mask = skip_layer_mask.reshape(batch_size, 1, 1)

        if (attention_mask is not None) and (not attn.use_tpu_flash_attention):
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)
        query = attn.q_norm(query)

        if encoder_hidden_states is not None:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            key = attn.to_k(encoder_hidden_states)
            key = attn.k_norm(key)
        else:  # if no context provided do self-attention
            encoder_hidden_states = hidden_states
            key = attn.to_k(hidden_states)
            key = attn.k_norm(key)
            if attn.use_rope:
                key = attn.apply_rotary_emb(key, freqs_cis)
                query = attn.apply_rotary_emb(query, freqs_cis)

        value = attn.to_v(encoder_hidden_states)
        value_for_stg = value

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)

        if attn.use_tpu_flash_attention:  # use tpu attention offload 'flash attention'
            q_segment_indexes = None
            if (
                attention_mask is not None
            ):  # if mask is required need to tune both segmenIds fields
                # attention_mask = torch.squeeze(attention_mask).to(torch.float32)
                attention_mask = attention_mask.to(torch.float32)
                q_segment_indexes = torch.ones(
                    batch_size, query.shape[2], device=query.device, dtype=torch.float32
                )
                assert (
                    attention_mask.shape[1] == key.shape[2]
                ), f"ERROR: KEY SHAPE must be same as attention mask [{key.shape[2]}, {attention_mask.shape[1]}]"

            assert (
                query.shape[2] % 128 == 0
            ), f"ERROR: QUERY SHAPE must be divisible by 128 (TPU limitation) [{query.shape[2]}]"
            assert (
                key.shape[2] % 128 == 0
            ), f"ERROR: KEY SHAPE must be divisible by 128 (TPU limitation) [{key.shape[2]}]"

            # run the TPU kernel implemented in jax with pallas
            hidden_states_a = flash_attention(
                q=query,
                k=key,
                v=value,
                q_segment_ids=q_segment_indexes,
                kv_segment_ids=attention_mask,
                sm_scale=attn.scale,
            )
        else:
            hidden_states_a = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        hidden_states_a = hidden_states_a.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states_a = hidden_states_a.to(query.dtype)

        if (
            skip_layer_mask is not None
            and skip_layer_strategy == SkipLayerStrategy.AttentionSkip
        ):
            hidden_states = hidden_states_a * skip_layer_mask + hidden_states * (
                1.0 - skip_layer_mask
            )
        elif (
            skip_layer_mask is not None
            and skip_layer_strategy == SkipLayerStrategy.AttentionValues
        ):
            hidden_states = hidden_states_a * skip_layer_mask + value_for_stg * (
                1.0 - skip_layer_mask
            )
        else:
            hidden_states = hidden_states_a

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
            if (
                skip_layer_mask is not None
                and skip_layer_strategy == SkipLayerStrategy.Residual
            ):
                skip_layer_mask = skip_layer_mask.reshape(batch_size, 1, 1, 1)

        if attn.residual_connection:
            if (
                skip_layer_mask is not None
                and skip_layer_strategy == SkipLayerStrategy.Residual
            ):
                hidden_states = hidden_states + residual * skip_layer_mask
            else:
                hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        query = attn.q_norm(query)
        key = attn.k_norm(key)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = nn.Linear

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(linear_cls(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        compatible_cls = (GEGLU, LoRACompatibleLinear)
        for module in self.net:
            if isinstance(module, compatible_cls):
                hidden_states = module(hidden_states, scale)
            else:
                hidden_states = module(hidden_states)
        return hidden_states
