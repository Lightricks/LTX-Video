# Adapted from: https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/models/transformers/transformer_2d.py
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.utils import BaseOutput, is_torch_version
from diffusers.utils import logging
from torch import nn

from ltx_video.models.transformers.attention import BasicTransformerBlock
from ltx_video.models.transformers.embeddings import get_3d_sincos_pos_embed

logger = logging.get_logger(__name__)


@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class Transformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        num_vector_embeds: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        adaptive_norm: str = "single_scale_shift",  # 'single_scale_shift' or 'single_scale'
        standardization_norm: str = "layer_norm",  # 'layer_norm' or 'rms_norm'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        project_to_2d_pos: bool = False,
        use_tpu_flash_attention: bool = False,  # if True uses the TPU attention offload ('flash attention')
        qk_norm: Optional[str] = None,
        positional_embedding_type: str = "absolute",
        positional_embedding_theta: Optional[float] = None,
        positional_embedding_max_pos: Optional[List[int]] = None,
        timestep_scale_multiplier: Optional[float] = None,
    ):
        super().__init__()
        self.use_tpu_flash_attention = (
            use_tpu_flash_attention  # FIXME: push config down to the attention modules
        )
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        self.project_to_2d_pos = project_to_2d_pos

        self.patchify_proj = nn.Linear(in_channels, inner_dim, bias=True)

        self.positional_embedding_type = positional_embedding_type
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.use_rope = self.positional_embedding_type == "rope"
        self.timestep_scale_multiplier = timestep_scale_multiplier

        if self.positional_embedding_type == "absolute":
            embed_dim_3d = (
                math.ceil((inner_dim / 2) * 3) if project_to_2d_pos else inner_dim
            )
            if self.project_to_2d_pos:
                self.to_2d_proj = torch.nn.Linear(embed_dim_3d, inner_dim, bias=False)
                self._init_to_2d_proj_weights(self.to_2d_proj)
        elif self.positional_embedding_type == "rope":
            if positional_embedding_theta is None:
                raise ValueError(
                    "If `positional_embedding_type` type is rope, `positional_embedding_theta` must also be defined"
                )
            if positional_embedding_max_pos is None:
                raise ValueError(
                    "If `positional_embedding_type` type is rope, `positional_embedding_max_pos` must also be defined"
                )

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    adaptive_norm=adaptive_norm,
                    standardization_norm=standardization_norm,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    use_tpu_flash_attention=use_tpu_flash_attention,
                    qk_norm=qk_norm,
                    use_rope=self.use_rope,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, inner_dim) / inner_dim**0.5
        )
        self.proj_out = nn.Linear(inner_dim, self.out_channels)

        self.adaln_single = AdaLayerNormSingle(
            inner_dim, use_additional_conditions=False
        )
        if adaptive_norm == "single_scale":
            self.adaln_single.linear = nn.Linear(inner_dim, 4 * inner_dim, bias=True)

        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels, hidden_size=inner_dim
            )

        self.gradient_checkpointing = False

    def set_use_tpu_flash_attention(self):
        r"""
        Function sets the flag in this object and propagates down the children. The flag will enforce the usage of TPU
        attention kernel.
        """
        logger.info("ENABLE TPU FLASH ATTENTION -> TRUE")
        self.use_tpu_flash_attention = True
        # push config down to the attention modules
        for block in self.transformer_blocks:
            block.set_use_tpu_flash_attention()

    def initialize(self, embedding_std: float, mode: Literal["ltx_video", "legacy"]):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(
            self.adaln_single.emb.timestep_embedder.linear_1.weight, std=embedding_std
        )
        nn.init.normal_(
            self.adaln_single.emb.timestep_embedder.linear_2.weight, std=embedding_std
        )
        nn.init.normal_(self.adaln_single.linear.weight, std=embedding_std)

        if hasattr(self.adaln_single.emb, "resolution_embedder"):
            nn.init.normal_(
                self.adaln_single.emb.resolution_embedder.linear_1.weight,
                std=embedding_std,
            )
            nn.init.normal_(
                self.adaln_single.emb.resolution_embedder.linear_2.weight,
                std=embedding_std,
            )
        if hasattr(self.adaln_single.emb, "aspect_ratio_embedder"):
            nn.init.normal_(
                self.adaln_single.emb.aspect_ratio_embedder.linear_1.weight,
                std=embedding_std,
            )
            nn.init.normal_(
                self.adaln_single.emb.aspect_ratio_embedder.linear_2.weight,
                std=embedding_std,
            )

        # Initialize caption embedding MLP:
        nn.init.normal_(self.caption_projection.linear_1.weight, std=embedding_std)
        nn.init.normal_(self.caption_projection.linear_1.weight, std=embedding_std)

        for block in self.transformer_blocks:
            if mode.lower() == "ltx_video":
                nn.init.constant_(block.attn1.to_out[0].weight, 0)
                nn.init.constant_(block.attn1.to_out[0].bias, 0)

            nn.init.constant_(block.attn2.to_out[0].weight, 0)
            nn.init.constant_(block.attn2.to_out[0].bias, 0)

            if mode.lower() == "ltx_video":
                nn.init.constant_(block.ff.net[2].weight, 0)
                nn.init.constant_(block.ff.net[2].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @staticmethod
    def _init_to_2d_proj_weights(linear_layer):
        input_features = linear_layer.weight.data.size(1)
        output_features = linear_layer.weight.data.size(0)

        # Start with a zero matrix
        identity_like = torch.zeros((output_features, input_features))

        # Fill the diagonal with 1's as much as possible
        min_features = min(output_features, input_features)
        identity_like[:min_features, :min_features] = torch.eye(min_features)
        linear_layer.weight.data = identity_like.to(linear_layer.weight.data.device)

    def get_fractional_positions(self, indices_grid):
        fractional_positions = torch.stack(
            [
                indices_grid[:, i] / self.positional_embedding_max_pos[i]
                for i in range(3)
            ],
            dim=-1,
        )
        return fractional_positions

    def precompute_freqs_cis(self, indices_grid, spacing="exp"):
        dtype = torch.float32  # We need full precision in the freqs_cis computation.
        dim = self.inner_dim
        theta = self.positional_embedding_theta

        fractional_positions = self.get_fractional_positions(indices_grid)

        start = 1
        end = theta
        device = fractional_positions.device
        if spacing == "exp":
            indices = theta ** (
                torch.linspace(
                    math.log(start, theta),
                    math.log(end, theta),
                    dim // 6,
                    device=device,
                    dtype=dtype,
                )
            )
            indices = indices.to(dtype=dtype)
        elif spacing == "exp_2":
            indices = 1.0 / theta ** (torch.arange(0, dim, 6, device=device) / dim)
            indices = indices.to(dtype=dtype)
        elif spacing == "linear":
            indices = torch.linspace(start, end, dim // 6, device=device, dtype=dtype)
        elif spacing == "sqrt":
            indices = torch.linspace(
                start**2, end**2, dim // 6, device=device, dtype=dtype
            ).sqrt()

        indices = indices * math.pi / 2

        if spacing == "exp_2":
            freqs = (
                (indices * fractional_positions.unsqueeze(-1))
                .transpose(-1, -2)
                .flatten(2)
            )
        else:
            freqs = (
                (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
                .transpose(-1, -2)
                .flatten(2)
            )

        cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
        if dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freq[:, :, : dim % 6])
            sin_padding = torch.zeros_like(cos_freq[:, :, : dim % 6])
            cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
            sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
        return cos_freq.to(self.dtype), sin_freq.to(self.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        indices_grid: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            indices_grid (`torch.LongTensor` of shape `(batch size, 3, num latent pixels)`):
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # for tpu attention offload 2d token masks are used. No need to transform.
        if not self.use_tpu_flash_attention:
            # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
            #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
            #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
            # expects mask of shape:
            #   [batch, key_tokens]
            # adds singleton query_tokens dimension:
            #   [batch,                    1, key_tokens]
            # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
            #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
            #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
            if attention_mask is not None and attention_mask.ndim == 2:
                # assume that mask is expressed as:
                #   (1 = keep,      0 = discard)
                # convert mask into a bias that can be added to attention scores:
                #       (keep = +0,     discard = -10000.0)
                attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (
                    1 - encoder_attention_mask.to(hidden_states.dtype)
                ) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        hidden_states = self.patchify_proj(hidden_states)

        if self.timestep_scale_multiplier:
            timestep = self.timestep_scale_multiplier * timestep

        if self.positional_embedding_type == "absolute":
            pos_embed_3d = self.get_absolute_pos_embed(indices_grid).to(
                hidden_states.device
            )
            if self.project_to_2d_pos:
                pos_embed = self.to_2d_proj(pos_embed_3d)
            hidden_states = (hidden_states + pos_embed).to(hidden_states.dtype)
            freqs_cis = None
        elif self.positional_embedding_type == "rope":
            freqs_cis = self.precompute_freqs_cis(indices_grid)

        batch_size = hidden_states.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    freqs_cis,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    freqs_cis=freqs_cis,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        if not return_dict:
            return (hidden_states,)

        return Transformer3DModelOutput(sample=hidden_states)

    def get_absolute_pos_embed(self, grid):
        grid_np = grid[0].cpu().numpy()
        embed_dim_3d = (
            math.ceil((self.inner_dim / 2) * 3)
            if self.project_to_2d_pos
            else self.inner_dim
        )
        pos_embed = get_3d_sincos_pos_embed(  # (f h w)
            embed_dim_3d,
            grid_np,
            h=int(max(grid_np[1]) + 1),
            w=int(max(grid_np[2]) + 1),
            f=int(max(grid_np[0] + 1)),
        )
        return torch.from_numpy(pos_embed).float().unsqueeze(0)
