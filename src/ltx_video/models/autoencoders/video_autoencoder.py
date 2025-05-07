import json
import os
from functools import partial
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Tuple, Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional

from diffusers.utils import logging

from ltx_video.utils.torch_utils import Identity
from ltx_video.models.autoencoders.conv_nd_factory import make_conv_nd, make_linear_nd
from ltx_video.models.autoencoders.pixel_norm import PixelNorm
from ltx_video.models.autoencoders.vae import AutoencoderKLWrapper

logger = logging.get_logger(__name__)


class VideoAutoencoder(AutoencoderKLWrapper):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *args,
        **kwargs,
    ):
        config_local_path = pretrained_model_name_or_path / "config.json"
        config = cls.load_config(config_local_path, **kwargs)
        video_vae = cls.from_config(config)
        video_vae.to(kwargs["torch_dtype"])

        model_local_path = pretrained_model_name_or_path / "autoencoder.pth"
        ckpt_state_dict = torch.load(model_local_path)
        video_vae.load_state_dict(ckpt_state_dict)

        statistics_local_path = (
            pretrained_model_name_or_path / "per_channel_statistics.json"
        )
        if statistics_local_path.exists():
            with open(statistics_local_path, "r") as file:
                data = json.load(file)
            transposed_data = list(zip(*data["data"]))
            data_dict = {
                col: torch.tensor(vals)
                for col, vals in zip(data["columns"], transposed_data)
            }
            video_vae.register_buffer("std_of_means", data_dict["std-of-means"])
            video_vae.register_buffer(
                "mean_of_means",
                data_dict.get(
                    "mean-of-means", torch.zeros_like(data_dict["std-of-means"])
                ),
            )

        return video_vae

    @staticmethod
    def from_config(config):
        assert (
            config["_class_name"] == "VideoAutoencoder"
        ), "config must have _class_name=VideoAutoencoder"
        if isinstance(config["dims"], list):
            config["dims"] = tuple(config["dims"])

        assert config["dims"] in [2, 3, (2, 1)], "dims must be 2, 3 or (2, 1)"

        double_z = config.get("double_z", True)
        latent_log_var = config.get(
            "latent_log_var", "per_channel" if double_z else "none"
        )
        use_quant_conv = config.get("use_quant_conv", True)

        if use_quant_conv and latent_log_var == "uniform":
            raise ValueError("uniform latent_log_var requires use_quant_conv=False")

        encoder = Encoder(
            dims=config["dims"],
            in_channels=config.get("in_channels", 3),
            out_channels=config["latent_channels"],
            block_out_channels=config["block_out_channels"],
            patch_size=config.get("patch_size", 1),
            latent_log_var=latent_log_var,
            norm_layer=config.get("norm_layer", "group_norm"),
            patch_size_t=config.get("patch_size_t", config.get("patch_size", 1)),
            add_channel_padding=config.get("add_channel_padding", False),
        )

        decoder = Decoder(
            dims=config["dims"],
            in_channels=config["latent_channels"],
            out_channels=config.get("out_channels", 3),
            block_out_channels=config["block_out_channels"],
            patch_size=config.get("patch_size", 1),
            norm_layer=config.get("norm_layer", "group_norm"),
            patch_size_t=config.get("patch_size_t", config.get("patch_size", 1)),
            add_channel_padding=config.get("add_channel_padding", False),
        )

        dims = config["dims"]
        return VideoAutoencoder(
            encoder=encoder,
            decoder=decoder,
            latent_channels=config["latent_channels"],
            dims=dims,
            use_quant_conv=use_quant_conv,
        )

    @property
    def config(self):
        return SimpleNamespace(
            _class_name="VideoAutoencoder",
            dims=self.dims,
            in_channels=self.encoder.conv_in.in_channels
            // (self.encoder.patch_size_t * self.encoder.patch_size**2),
            out_channels=self.decoder.conv_out.out_channels
            // (self.decoder.patch_size_t * self.decoder.patch_size**2),
            latent_channels=self.decoder.conv_in.in_channels,
            block_out_channels=[
                self.encoder.down_blocks[i].res_blocks[-1].conv1.out_channels
                for i in range(len(self.encoder.down_blocks))
            ],
            scaling_factor=1.0,
            norm_layer=self.encoder.norm_layer,
            patch_size=self.encoder.patch_size,
            latent_log_var=self.encoder.latent_log_var,
            use_quant_conv=self.use_quant_conv,
            patch_size_t=self.encoder.patch_size_t,
            add_channel_padding=self.encoder.add_channel_padding,
        )

    @property
    def is_video_supported(self):
        """
        Check if the model supports video inputs of shape (B, C, F, H, W). Otherwise, the model only supports 2D images.
        """
        return self.dims != 2

    @property
    def downscale_factor(self):
        return self.encoder.downsample_factor

    def to_json_string(self) -> str:
        import json

        return json.dumps(self.config.__dict__)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        model_keys = set(name for name, _ in self.named_parameters())

        key_mapping = {
            ".resnets.": ".res_blocks.",
            "downsamplers.0": "downsample",
            "upsamplers.0": "upsample",
        }

        converted_state_dict = {}
        for key, value in state_dict.items():
            for k, v in key_mapping.items():
                key = key.replace(k, v)

            if "norm" in key and key not in model_keys:
                logger.info(
                    f"Removing key {key} from state_dict as it is not present in the model"
                )
                continue

            converted_state_dict[key] = value

        super().load_state_dict(converted_state_dict, strict=strict)

    def last_layer(self):
        if hasattr(self.decoder, "conv_out"):
            if isinstance(self.decoder.conv_out, nn.Sequential):
                last_layer = self.decoder.conv_out[-1]
            else:
                last_layer = self.decoder.conv_out
        else:
            last_layer = self.decoder.layers[-1]
        return last_layer


class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        patch_size (`int`, *optional*, defaults to 1):
            The patch size to use. Should be a power of 2.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        latent_log_var (`str`, *optional*, defaults to `per_channel`):
            The number of channels for the log variance. Can be either `per_channel`, `uniform`, or `none`.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]] = 3,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        patch_size: Union[int, Tuple[int]] = 1,
        norm_layer: str = "group_norm",  # group_norm, pixel_norm
        latent_log_var: str = "per_channel",
        patch_size_t: Optional[int] = None,
        add_channel_padding: Optional[bool] = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t if patch_size_t is not None else patch_size
        self.add_channel_padding = add_channel_padding
        self.layers_per_block = layers_per_block
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        if add_channel_padding:
            in_channels = in_channels * self.patch_size**3
        else:
            in_channels = in_channels * self.patch_size_t * self.patch_size**2
        self.in_channels = in_channels
        output_channel = block_out_channels[0]

        self.conv_in = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownEncoderBlock3D(
                dims=dims,
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block,
                add_downsample=not is_final_block and 2**i >= patch_size,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_groups=norm_num_groups,
                norm_layer=norm_layer,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock3D(
            dims=dims,
            in_channels=block_out_channels[-1],
            num_layers=self.layers_per_block,
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
        )

        # out
        if norm_layer == "group_norm":
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[-1],
                num_groups=norm_num_groups,
                eps=1e-6,
            )
        elif norm_layer == "pixel_norm":
            self.conv_norm_out = PixelNorm()
        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == "per_channel":
            conv_out_channels *= 2
        elif latent_log_var == "uniform":
            conv_out_channels += 1
        elif latent_log_var != "none":
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")
        self.conv_out = make_conv_nd(
            dims, block_out_channels[-1], conv_out_channels, 3, padding=1
        )

        self.gradient_checkpointing = False

    @property
    def downscale_factor(self):
        return (
            2
            ** len(
                [
                    block
                    for block in self.down_blocks
                    if isinstance(block.downsample, Downsample3D)
                ]
            )
            * self.patch_size
        )

    def forward(
        self, sample: torch.FloatTensor, return_features=False
    ) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""

        downsample_in_time = sample.shape[2] != 1

        # patchify
        patch_size_t = self.patch_size_t if downsample_in_time else 1
        sample = patchify(
            sample,
            patch_size_hw=self.patch_size,
            patch_size_t=patch_size_t,
            add_channel_padding=self.add_channel_padding,
        )

        sample = self.conv_in(sample)

        checkpoint_fn = (
            partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else lambda x: x
        )

        if return_features:
            features = []
        for down_block in self.down_blocks:
            sample = checkpoint_fn(down_block)(
                sample, downsample_in_time=downsample_in_time
            )
            if return_features:
                features.append(sample)

        sample = checkpoint_fn(self.mid_block)(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.latent_log_var == "uniform":
            last_channel = sample[:, -1:, ...]
            num_dims = sample.dim()

            if num_dims == 4:
                # For shape (B, C, H, W)
                repeated_last_channel = last_channel.repeat(
                    1, sample.shape[1] - 2, 1, 1
                )
                sample = torch.cat([sample, repeated_last_channel], dim=1)
            elif num_dims == 5:
                # For shape (B, C, F, H, W)
                repeated_last_channel = last_channel.repeat(
                    1, sample.shape[1] - 2, 1, 1, 1
                )
                sample = torch.cat([sample, repeated_last_channel], dim=1)
            else:
                raise ValueError(f"Invalid input shape: {sample.shape}")

        if return_features:
            features.append(sample[:, : self.latent_channels, ...])
            return sample, features
        return sample


class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        patch_size (`int`, *optional*, defaults to 1):
            The patch size to use. Should be a power of 2.
        norm_layer (`str`, *optional*, defaults to `group_norm`):
            The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
    """

    def __init__(
        self,
        dims,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        patch_size: int = 1,
        norm_layer: str = "group_norm",
        patch_size_t: Optional[int] = None,
        add_channel_padding: Optional[bool] = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t if patch_size_t is not None else patch_size
        self.add_channel_padding = add_channel_padding
        self.layers_per_block = layers_per_block
        if add_channel_padding:
            out_channels = out_channels * self.patch_size**3
        else:
            out_channels = out_channels * self.patch_size_t * self.patch_size**2
        self.out_channels = out_channels

        self.conv_in = make_conv_nd(
            dims,
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        self.mid_block = UNetMidBlock3D(
            dims=dims,
            in_channels=block_out_channels[-1],
            num_layers=self.layers_per_block,
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpDecoderBlock3D(
                dims=dims,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block
                and 2 ** (len(block_out_channels) - i - 1) > patch_size,
                resnet_eps=1e-6,
                resnet_groups=norm_num_groups,
                norm_layer=norm_layer,
            )
            self.up_blocks.append(up_block)

        if norm_layer == "group_norm":
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
            )
        elif norm_layer == "pixel_norm":
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims, block_out_channels[0], out_channels, 3, padding=1
        )

        self.gradient_checkpointing = False

    def forward(self, sample: torch.FloatTensor, target_shape) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""
        assert target_shape is not None, "target_shape must be provided"
        upsample_in_time = sample.shape[2] < target_shape[2]

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        checkpoint_fn = (
            partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else lambda x: x
        )

        sample = checkpoint_fn(self.mid_block)(sample)
        sample = sample.to(upscale_dtype)

        for up_block in self.up_blocks:
            sample = checkpoint_fn(up_block)(sample, upsample_in_time=upsample_in_time)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # un-patchify
        patch_size_t = self.patch_size_t if upsample_in_time else 1
        sample = unpatchify(
            sample,
            patch_size_hw=self.patch_size,
            patch_size_t=patch_size_t,
            add_channel_padding=self.add_channel_padding,
        )

        return sample


class DownEncoderBlock3D(nn.Module):
    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        norm_layer: str = "group_norm",
    ):
        super().__init__()
        res_blocks = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            res_blocks.append(
                ResnetBlock3D(
                    dims=dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    norm_layer=norm_layer,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)

        if add_downsample:
            self.downsample = Downsample3D(
                dims,
                out_channels,
                out_channels=out_channels,
                padding=downsample_padding,
            )
        else:
            self.downsample = Identity()

    def forward(
        self, hidden_states: torch.FloatTensor, downsample_in_time
    ) -> torch.FloatTensor:
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states)

        hidden_states = self.downsample(
            hidden_states, downsample_in_time=downsample_in_time
        )

        return hidden_states


class UNetMidBlock3D(nn.Module):
    """
    A 3D UNet mid-block [`UNetMidBlock3D`] with multiple residual blocks.

    Args:
        in_channels (`int`): The number of input channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.

    Returns:
        `torch.FloatTensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.

    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        norm_layer: str = "group_norm",
    ):
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )

        self.res_blocks = nn.ModuleList(
            [
                ResnetBlock3D(
                    dims=dims,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states)

        return hidden_states


class UpDecoderBlock3D(nn.Module):
    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        add_upsample: bool = True,
        norm_layer: str = "group_norm",
    ):
        super().__init__()
        res_blocks = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            res_blocks.append(
                ResnetBlock3D(
                    dims=dims,
                    in_channels=input_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    norm_layer=norm_layer,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)

        if add_upsample:
            self.upsample = Upsample3D(
                dims=dims, channels=out_channels, out_channels=out_channels
            )
        else:
            self.upsample = Identity()

        self.resolution_idx = resolution_idx

    def forward(
        self, hidden_states: torch.FloatTensor, upsample_in_time=True
    ) -> torch.FloatTensor:
        for resnet in self.res_blocks:
            hidden_states = resnet(hidden_states)

        hidden_states = self.upsample(hidden_states, upsample_in_time=upsample_in_time)

        return hidden_states


class ResnetBlock3D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    def __init__(
        self,
        dims: Union[int, Tuple[int, int]],
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        norm_layer: str = "group_norm",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if norm_layer == "group_norm":
            self.norm1 = torch.nn.GroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )
        elif norm_layer == "pixel_norm":
            self.norm1 = PixelNorm()

        self.non_linearity = nn.SiLU()

        self.conv1 = make_conv_nd(
            dims, in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if norm_layer == "group_norm":
            self.norm2 = torch.nn.GroupNorm(
                num_groups=groups, num_channels=out_channels, eps=eps, affine=True
            )
        elif norm_layer == "pixel_norm":
            self.norm2 = PixelNorm()

        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = make_conv_nd(
            dims, out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.conv_shortcut = (
            make_linear_nd(
                dims=dims, in_channels=in_channels, out_channels=out_channels
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        input_tensor: torch.FloatTensor,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)

        hidden_states = self.non_linearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)

        hidden_states = self.non_linearity(hidden_states)

        hidden_states = self.dropout(hidden_states)

        hidden_states = self.conv2(hidden_states)

        input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


class Downsample3D(nn.Module):
    def __init__(
        self,
        dims,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        stride: int = 2
        self.padding = padding
        self.in_channels = in_channels
        self.dims = dims
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, downsample_in_time=True):
        conv = self.conv
        if self.padding == 0:
            if self.dims == 2:
                padding = (0, 1, 0, 1)
            else:
                padding = (0, 1, 0, 1, 0, 1 if downsample_in_time else 0)

            x = functional.pad(x, padding, mode="constant", value=0)

            if self.dims == (2, 1) and not downsample_in_time:
                return conv(x, skip_time_conv=True)

        return conv(x)


class Upsample3D(nn.Module):
    """
    An upsampling layer for 3D tensors of shape (B, C, D, H, W).

    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, dims, channels, out_channels=None):
        super().__init__()
        self.dims = dims
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = make_conv_nd(
            dims, channels, out_channels, kernel_size=3, padding=1, bias=True
        )

    def forward(self, x, upsample_in_time):
        if self.dims == 2:
            x = functional.interpolate(
                x, (x.shape[2] * 2, x.shape[3] * 2), mode="nearest"
            )
        else:
            time_scale_factor = 2 if upsample_in_time else 1
            # print("before:", x.shape)
            b, c, d, h, w = x.shape
            x = rearrange(x, "b c d h w -> (b d) c h w")
            # height and width interpolate
            x = functional.interpolate(
                x, (x.shape[2] * 2, x.shape[3] * 2), mode="nearest"
            )
            _, _, h, w = x.shape

            if not upsample_in_time and self.dims == (2, 1):
                x = rearrange(x, "(b d) c h w -> b c d h w ", b=b, h=h, w=w)
                return self.conv(x, skip_time_conv=True)

            # Second ** upsampling ** which is essentially treated as a 1D convolution across the 'd' dimension
            x = rearrange(x, "(b d) c h w -> (b h w) c 1 d", b=b)

            # (b h w) c 1 d
            new_d = x.shape[-1] * time_scale_factor
            x = functional.interpolate(x, (1, new_d), mode="nearest")
            # (b h w) c 1 new_d
            x = rearrange(
                x, "(b h w) c 1 new_d  -> b c new_d h w", b=b, h=h, w=w, new_d=new_d
            )
            # b c d h w

            # x = functional.interpolate(
            #     x, (x.shape[2] * time_scale_factor, x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            # )
            # print("after:", x.shape)

        return self.conv(x)


def patchify(x, patch_size_hw, patch_size_t=1, add_channel_padding=False):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(
            x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw
        )
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    if (
        (x.dim() == 5)
        and (patch_size_hw > patch_size_t)
        and (patch_size_t > 1 or add_channel_padding)
    ):
        channels_to_pad = x.shape[1] * (patch_size_hw // patch_size_t) - x.shape[1]
        padding_zeros = torch.zeros(
            x.shape[0],
            channels_to_pad,
            x.shape[2],
            x.shape[3],
            x.shape[4],
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([padding_zeros, x], dim=1)

    return x


def unpatchify(x, patch_size_hw, patch_size_t=1, add_channel_padding=False):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if (
        (x.dim() == 5)
        and (patch_size_hw > patch_size_t)
        and (patch_size_t > 1 or add_channel_padding)
    ):
        channels_to_keep = int(x.shape[1] * (patch_size_t / patch_size_hw))
        x = x[:, :channels_to_keep, :, :, :]

    if x.dim() == 4:
        x = rearrange(
            x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw
        )
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )

    return x


def create_video_autoencoder_config(
    latent_channels: int = 4,
):
    config = {
        "_class_name": "VideoAutoencoder",
        "dims": (
            2,
            1,
        ),  # 2 for Conv2, 3 for Conv3d, (2, 1) for Conv2d followed by Conv1d
        "in_channels": 3,  # Number of input color channels (e.g., RGB)
        "out_channels": 3,  # Number of output color channels
        "latent_channels": latent_channels,  # Number of channels in the latent space representation
        "block_out_channels": [
            128,
            256,
            512,
            512,
        ],  # Number of output channels of each encoder / decoder inner block
        "patch_size": 1,
    }

    return config


def create_video_autoencoder_pathify4x4x4_config(
    latent_channels: int = 4,
):
    config = {
        "_class_name": "VideoAutoencoder",
        "dims": (
            2,
            1,
        ),  # 2 for Conv2, 3 for Conv3d, (2, 1) for Conv2d followed by Conv1d
        "in_channels": 3,  # Number of input color channels (e.g., RGB)
        "out_channels": 3,  # Number of output color channels
        "latent_channels": latent_channels,  # Number of channels in the latent space representation
        "block_out_channels": [512]
        * 4,  # Number of output channels of each encoder / decoder inner block
        "patch_size": 4,
        "latent_log_var": "uniform",
    }

    return config


def create_video_autoencoder_pathify4x4_config(
    latent_channels: int = 4,
):
    config = {
        "_class_name": "VideoAutoencoder",
        "dims": 2,  # 2 for Conv2, 3 for Conv3d, (2, 1) for Conv2d followed by Conv1d
        "in_channels": 3,  # Number of input color channels (e.g., RGB)
        "out_channels": 3,  # Number of output color channels
        "latent_channels": latent_channels,  # Number of channels in the latent space representation
        "block_out_channels": [512]
        * 4,  # Number of output channels of each encoder / decoder inner block
        "patch_size": 4,
        "norm_layer": "pixel_norm",
    }

    return config


def test_vae_patchify_unpatchify():
    import torch

    x = torch.randn(2, 3, 8, 64, 64)
    x_patched = patchify(x, patch_size_hw=4, patch_size_t=4)
    x_unpatched = unpatchify(x_patched, patch_size_hw=4, patch_size_t=4)
    assert torch.allclose(x, x_unpatched)


def demo_video_autoencoder_forward_backward():
    # Configuration for the VideoAutoencoder
    config = create_video_autoencoder_pathify4x4x4_config()

    # Instantiate the VideoAutoencoder with the specified configuration
    video_autoencoder = VideoAutoencoder.from_config(config)

    print(video_autoencoder)

    # Print the total number of parameters in the video autoencoder
    total_params = sum(p.numel() for p in video_autoencoder.parameters())
    print(f"Total number of parameters in VideoAutoencoder: {total_params:,}")

    # Create a mock input tensor simulating a batch of videos
    # Shape: (batch_size, channels, depth, height, width)
    # E.g., 4 videos, each with 3 color channels, 16 frames, and 64x64 pixels per frame
    input_videos = torch.randn(2, 3, 8, 64, 64)

    # Forward pass: encode and decode the input videos
    latent = video_autoencoder.encode(input_videos).latent_dist.mode()
    print(f"input shape={input_videos.shape}")
    print(f"latent shape={latent.shape}")
    reconstructed_videos = video_autoencoder.decode(
        latent, target_shape=input_videos.shape
    ).sample

    print(f"reconstructed shape={reconstructed_videos.shape}")

    # Calculate the loss (e.g., mean squared error)
    loss = torch.nn.functional.mse_loss(input_videos, reconstructed_videos)

    # Perform backward pass
    loss.backward()

    print(f"Demo completed with loss: {loss.item()}")


# Ensure to call the demo function to execute the forward and backward pass
if __name__ == "__main__":
    demo_video_autoencoder_forward_backward()
