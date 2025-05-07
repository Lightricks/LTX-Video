from typing import Tuple
import torch
from diffusers import AutoencoderKL
from einops import rearrange
from torch import Tensor


from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.video_autoencoder import (
    Downsample3D,
    VideoAutoencoder,
)

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


def vae_encode(
    media_items: Tensor,
    vae: AutoencoderKL,
    split_size: int = 1,
    vae_per_channel_normalize=False,
) -> Tensor:
    """
    Encodes media items (images or videos) into latent representations using a specified VAE model.
    The function supports processing batches of images or video frames and can handle the processing
    in smaller sub-batches if needed.

    Args:
        media_items (Tensor): A torch Tensor containing the media items to encode. The expected
            shape is (batch_size, channels, height, width) for images or (batch_size, channels,
            frames, height, width) for videos.
        vae (AutoencoderKL): An instance of the `AutoencoderKL` class from the `diffusers` library,
            pre-configured and loaded with the appropriate model weights.
        split_size (int, optional): The number of sub-batches to split the input batch into for encoding.
            If set to more than 1, the input media items are processed in smaller batches according to
            this value. Defaults to 1, which processes all items in a single batch.

    Returns:
        Tensor: A torch Tensor of the encoded latent representations. The shape of the tensor is adjusted
            to match the input shape, scaled by the model's configuration.

    Examples:
        >>> import torch
        >>> from diffusers import AutoencoderKL
        >>> vae = AutoencoderKL.from_pretrained('your-model-name')
        >>> images = torch.rand(10, 3, 8 256, 256)  # Example tensor with 10 videos of 8 frames.
        >>> latents = vae_encode(images, vae)
        >>> print(latents.shape)  # Output shape will depend on the model's latent configuration.

    Note:
        In case of a video, the function encodes the media item frame-by frame.
    """
    is_video_shaped = media_items.dim() == 5
    batch_size, channels = media_items.shape[0:2]

    if channels != 3:
        raise ValueError(f"Expects tensors with 3 channels, got {channels}.")

    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        media_items = rearrange(media_items, "b c n h w -> (b n) c h w")
    if split_size > 1:
        if len(media_items) % split_size != 0:
            raise ValueError(
                "Error: The batch size must be divisible by 'train.vae_bs_split"
            )
        encode_bs = len(media_items) // split_size
        # latents = [vae.encode(image_batch).latent_dist.sample() for image_batch in media_items.split(encode_bs)]
        latents = []
        if media_items.device.type == "xla":
            xm.mark_step()
        for image_batch in media_items.split(encode_bs):
            latents.append(vae.encode(image_batch).latent_dist.sample())
            if media_items.device.type == "xla":
                xm.mark_step()
        latents = torch.cat(latents, dim=0)
    else:
        latents = vae.encode(media_items).latent_dist.sample()

    latents = normalize_latents(latents, vae, vae_per_channel_normalize)
    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        latents = rearrange(latents, "(b n) c h w -> b c n h w", b=batch_size)
    return latents


def vae_decode(
    latents: Tensor,
    vae: AutoencoderKL,
    is_video: bool = True,
    split_size: int = 1,
    vae_per_channel_normalize=False,
    timestep=None,
) -> Tensor:
    is_video_shaped = latents.dim() == 5
    batch_size = latents.shape[0]

    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        latents = rearrange(latents, "b c n h w -> (b n) c h w")
    if split_size > 1:
        if len(latents) % split_size != 0:
            raise ValueError(
                "Error: The batch size must be divisible by 'train.vae_bs_split"
            )
        encode_bs = len(latents) // split_size
        image_batch = [
            _run_decoder(
                latent_batch, vae, is_video, vae_per_channel_normalize, timestep
            )
            for latent_batch in latents.split(encode_bs)
        ]
        images = torch.cat(image_batch, dim=0)
    else:
        images = _run_decoder(
            latents, vae, is_video, vae_per_channel_normalize, timestep
        )

    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        images = rearrange(images, "(b n) c h w -> b c n h w", b=batch_size)
    return images


def _run_decoder(
    latents: Tensor,
    vae: AutoencoderKL,
    is_video: bool,
    vae_per_channel_normalize=False,
    timestep=None,
) -> Tensor:
    if isinstance(vae, (VideoAutoencoder, CausalVideoAutoencoder)):
        *_, fl, hl, wl = latents.shape
        temporal_scale, spatial_scale, _ = get_vae_size_scale_factor(vae)
        latents = latents.to(vae.dtype)
        vae_decode_kwargs = {}
        if timestep is not None:
            vae_decode_kwargs["timestep"] = timestep
        image = vae.decode(
            un_normalize_latents(latents, vae, vae_per_channel_normalize),
            return_dict=False,
            target_shape=(
                1,
                3,
                fl * temporal_scale if is_video else 1,
                hl * spatial_scale,
                wl * spatial_scale,
            ),
            **vae_decode_kwargs,
        )[0]
    else:
        image = vae.decode(
            un_normalize_latents(latents, vae, vae_per_channel_normalize),
            return_dict=False,
        )[0]
    return image


def get_vae_size_scale_factor(vae: AutoencoderKL) -> float:
    if isinstance(vae, CausalVideoAutoencoder):
        spatial = vae.spatial_downscale_factor
        temporal = vae.temporal_downscale_factor
    else:
        down_blocks = len(
            [
                block
                for block in vae.encoder.down_blocks
                if isinstance(block.downsample, Downsample3D)
            ]
        )
        spatial = vae.config.patch_size * 2**down_blocks
        temporal = (
            vae.config.patch_size_t * 2**down_blocks
            if isinstance(vae, VideoAutoencoder)
            else 1
        )

    return (temporal, spatial, spatial)


def latent_to_pixel_coords(
    latent_coords: Tensor, vae: AutoencoderKL, causal_fix: bool = False
) -> Tensor:
    """
    Converts latent coordinates to pixel coordinates by scaling them according to the VAE's
    configuration.

    Args:
        latent_coords (Tensor): A tensor of shape [batch_size, 3, num_latents]
        containing the latent corner coordinates of each token.
        vae (AutoencoderKL): The VAE model
        causal_fix (bool): Whether to take into account the different temporal scale
            of the first frame. Default = False for backwards compatibility.
    Returns:
        Tensor: A tensor of pixel coordinates corresponding to the input latent coordinates.
    """

    scale_factors = get_vae_size_scale_factor(vae)
    causal_fix = isinstance(vae, CausalVideoAutoencoder) and causal_fix
    pixel_coords = latent_to_pixel_coords_from_factors(
        latent_coords, scale_factors, causal_fix
    )
    return pixel_coords


def latent_to_pixel_coords_from_factors(
    latent_coords: Tensor, scale_factors: Tuple, causal_fix: bool = False
) -> Tensor:
    pixel_coords = (
        latent_coords
        * torch.tensor(scale_factors, device=latent_coords.device)[None, :, None]
    )
    if causal_fix:
        # Fix temporal scale for first frame to 1 due to causality
        pixel_coords[:, 0] = (pixel_coords[:, 0] + 1 - scale_factors[0]).clamp(min=0)
    return pixel_coords


def normalize_latents(
    latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False
) -> Tensor:
    return (
        (latents - vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1))
        / vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        if vae_per_channel_normalize
        else latents * vae.config.scaling_factor
    )


def un_normalize_latents(
    latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False
) -> Tensor:
    return (
        latents * vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        + vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        if vae_per_channel_normalize
        else latents / vae.config.scaling_factor
    )
