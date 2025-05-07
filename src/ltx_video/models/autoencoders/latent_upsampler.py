from typing import Optional, Union
from pathlib import Path
import os
import json

import torch
import torch.nn as nn
from einops import rearrange
from diffusers import ConfigMixin, ModelMixin
from safetensors.torch import safe_open

from ltx_video.models.autoencoders.pixel_shuffle import PixelShuffleND


class ResBlock(nn.Module):
    def __init__(
        self, channels: int, mid_channels: Optional[int] = None, dims: int = 3
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        Conv = nn.Conv2d if dims == 2 else nn.Conv3d

        self.conv1 = Conv(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, mid_channels)
        self.conv2 = Conv(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x + residual)
        return x


class LatentUpsampler(ModelMixin, ConfigMixin):
    """
    Model to spatially upsample VAE latents.

    Args:
        in_channels (`int`): Number of channels in the input latent
        mid_channels (`int`): Number of channels in the middle layers
        num_blocks_per_stage (`int`): Number of ResBlocks to use in each stage (pre/post upsampling)
        dims (`int`): Number of dimensions for convolutions (2 or 3)
        spatial_upsample (`bool`): Whether to spatially upsample the latent
        temporal_upsample (`bool`): Whether to temporally upsample the latent
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        dims: int = 3,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.dims = dims
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample

        Conv = nn.Conv2d if dims == 2 else nn.Conv3d

        self.initial_conv = Conv(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(32, mid_channels)
        self.initial_activation = nn.SiLU()

        self.res_blocks = nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        if spatial_upsample and temporal_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv3d(mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(3),
            )
        elif spatial_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(2),
            )
        elif temporal_upsample:
            self.upsampler = nn.Sequential(
                nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(1),
            )
        else:
            raise ValueError(
                "Either spatial_upsample or temporal_upsample must be True"
            )

        self.post_upsample_res_blocks = nn.ModuleList(
            [ResBlock(mid_channels, dims=dims) for _ in range(num_blocks_per_stage)]
        )

        self.final_conv = Conv(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = latent.shape

        if self.dims == 2:
            x = rearrange(latent, "b c f h w -> (b f) c h w")
            x = self.initial_conv(x)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            x = self.upsampler(x)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)
        else:
            x = self.initial_conv(latent)
            x = self.initial_norm(x)
            x = self.initial_activation(x)

            for block in self.res_blocks:
                x = block(x)

            if self.temporal_upsample:
                x = self.upsampler(x)
                x = x[:, :, 1:, :, :]
            else:
                x = rearrange(x, "b c f h w -> (b f) c h w")
                x = self.upsampler(x)
                x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

            for block in self.post_upsample_res_blocks:
                x = block(x)

            x = self.final_conv(x)

        return x

    @classmethod
    def from_config(cls, config):
        return cls(
            in_channels=config.get("in_channels", 4),
            mid_channels=config.get("mid_channels", 128),
            num_blocks_per_stage=config.get("num_blocks_per_stage", 4),
            dims=config.get("dims", 2),
            spatial_upsample=config.get("spatial_upsample", True),
            temporal_upsample=config.get("temporal_upsample", False),
        )

    def config(self):
        return {
            "_class_name": "LatentUpsampler",
            "in_channels": self.in_channels,
            "mid_channels": self.mid_channels,
            "num_blocks_per_stage": self.num_blocks_per_stage,
            "dims": self.dims,
            "spatial_upsample": self.spatial_upsample,
            "temporal_upsample": self.temporal_upsample,
        }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[Union[str, os.PathLike]],
        *args,
        **kwargs,
    ):
        pretrained_model_path = Path(pretrained_model_path)
        if pretrained_model_path.is_file() and str(pretrained_model_path).endswith(
            ".safetensors"
        ):
            state_dict = {}
            with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            config = json.loads(metadata["config"])
            with torch.device("meta"):
                latent_upsampler = LatentUpsampler.from_config(config)
            latent_upsampler.load_state_dict(state_dict, assign=True)
        return latent_upsampler


if __name__ == "__main__":
    latent_upsampler = LatentUpsampler(num_blocks_per_stage=4, dims=3)
    print(latent_upsampler)
    total_params = sum(p.numel() for p in latent_upsampler.parameters())
    print(f"Total number of parameters: {total_params:,}")
    latent = torch.randn(1, 128, 9, 16, 16)
    upsampled_latent = latent_upsampler(latent)
    print(f"Upsampled latent shape: {upsampled_latent.shape}")
