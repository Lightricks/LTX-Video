import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging

from typing import Union
from typing import Optional

import imageio
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.conditioning_method import ConditioningMethod

MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257

g_vae=None
g_unet=None
g_scheduler=None
g_patchifier=None
g_text_encoder=None
g_tokenizer=None
g_pipeline=None

def load_vae(vae_dir: Path) -> CausalVideoAutoencoder:
    """
    Load a video autoencoder model (VAE) from a specified directory.

    The function reads the configuration file and model weights from the provided
    directory to initialize and return a `CausalVideoAutoencoder` instance. If a CUDA
    device is available, the model is moved to the GPU and converted to `bfloat16` precision.

    Args:
        vae_dir (Path): 
            A `Path` object representing the directory containing the VAE model files:
            - `vae_diffusion_pytorch_model.safetensors`: Model weights in safetensors format.
            - `config.json`: Configuration file for the VAE model.

    Returns:
        CausalVideoAutoencoder:
            A `CausalVideoAutoencoder` model initialized with the loaded weights and
            configuration, moved to GPU (if available) and set to `bfloat16` precision.

    Raises:
        FileNotFoundError: 
            If any of the required files (`vae_diffusion_pytorch_model.safetensors` 
            or `config.json`) are not found in the specified directory.

    Example:
        >>> from pathlib import Path
        >>> vae_model = load_vae(Path("models/vae"))
        >>> print(vae_model)
    """
    print("Loading VAE")
    vae_ckpt_path = vae_dir / "vae_diffusion_pytorch_model.safetensors"
    vae_config_path = vae_dir / "config.json"
    with open(vae_config_path, "r") as f:
        vae_config = json.load(f)
    vae = CausalVideoAutoencoder.from_config(vae_config)
    vae_state_dict = safetensors.torch.load_file(vae_ckpt_path)
    vae.load_state_dict(vae_state_dict)
    if torch.cuda.is_available():
        vae = vae.cuda()
    return vae.to(torch.bfloat16)

def load_unet(unet_dir: Path) -> Transformer3DModel:
    """
    Load a U-Net model for diffusion tasks from a specified directory.

    This function initializes a `Transformer3DModel` by reading its configuration and
    model weights from the provided directory. The weights are loaded from a safetensors
    file, and the configuration is loaded from a JSON file. If a CUDA device is available,
    the model is moved to the GPU.

    Args:
        unet_dir (Path): 
            A `Path` object representing the directory containing the U-Net model files:
            - `unet_diffusion_pytorch_model.safetensors`: Model weights in safetensors format.
            - `config.json`: Configuration file for the U-Net model.

    Returns:
        Transformer3DModel:
            A `Transformer3DModel` initialized with the loaded weights and configuration,
            moved to GPU (if available).

    Raises:
        FileNotFoundError: 
            If any of the required files (`unet_diffusion_pytorch_model.safetensors` 
            or `config.json`) are not found in the specified directory.
        RuntimeError: 
            If there is an issue loading the model weights or configuration.

    Example:
        >>> from pathlib import Path
        >>> unet_model = load_unet(Path("models/unet"))
        >>> print(unet_model)
    """
    print("Loading UNET")
    unet_ckpt_path = unet_dir / "unet_diffusion_pytorch_model.safetensors"
    unet_config_path = unet_dir / "config.json"
    transformer_config = Transformer3DModel.load_config(unet_config_path)
    transformer = Transformer3DModel.from_config(transformer_config)
    unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
    transformer.load_state_dict(unet_state_dict, strict=True)
    if torch.cuda.is_available():
        transformer = transformer.cuda()
    return transformer

def load_scheduler(scheduler_dir: Path) -> RectifiedFlowScheduler:
    """
    Load a scheduler for controlling the diffusion process from a specified directory.

    This function initializes a `RectifiedFlowScheduler` instance by loading its configuration
    from a JSON file. The scheduler is responsible for managing the rectified flow during the
    diffusion process.

    Args:
        scheduler_dir (Path): 
            A `Path` object representing the directory containing the scheduler configuration file:
            - `scheduler_config.json`: Configuration file for the scheduler.

    Returns:
        RectifiedFlowScheduler:
            A `RectifiedFlowScheduler` instance initialized with the loaded configuration.

    Raises:
        FileNotFoundError: 
            If the required file (`scheduler_config.json`) is not found in the specified directory.
        ValueError: 
            If the configuration file is invalid or cannot be parsed.

    Example:
        >>> from pathlib import Path
        >>> scheduler = load_scheduler(Path("models/scheduler"))
        >>> print(scheduler)
    """
    print("Loading Scheduler")
    scheduler_config_path = scheduler_dir / "scheduler_config.json"
    scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
    return RectifiedFlowScheduler.from_config(scheduler_config)

def load_image_to_tensor_with_resize_and_crop(
    image_path: Union[str, Path], target_height: int = 512, target_width: int = 768
) -> torch.Tensor:
    """
    Load an image from a file, resize and crop it to the specified dimensions, and convert it to a 5D tensor.

    This function performs the following steps:
    1. Loads the image from the given path and converts it to RGB format.
    2. Crops the image to match the target aspect ratio while preserving the center.
    3. Resizes the cropped image to the specified target dimensions.
    4. Normalizes pixel values to the range [-1, 1].
    5. Converts the image into a 5D tensor suitable for use in machine learning pipelines.

    Args:
        image_path (Union[str, Path]): 
            Path to the input image file.
        target_height (int, optional): 
            Desired height of the output image. Defaults to 512.
        target_width (int, optional): 
            Desired width of the output image. Defaults to 768.

    Returns:
        torch.Tensor:
            A 5D tensor of shape `(1, 3, 1, target_height, target_width)` representing the image, where:
            - The first dimension is the batch size (set to 1).
            - The second dimension is the number of color channels (RGB, so 3).
            - The third dimension is the number of frames (set to 1 for a single image).
            - The last two dimensions are the height and width of the image.

    Raises:
        FileNotFoundError: 
            If the image file is not found at the specified path.
        ValueError: 
            If the image cannot be loaded or processed.

    Example:
        >>> from pathlib import Path
        >>> image_tensor = load_image_to_tensor_with_resize_and_crop(Path("input.jpg"), 512, 768)
        >>> print(image_tensor.shape)  # Output: torch.Size([1, 3, 1, 512, 768])
    """
    image = Image.open(image_path).convert("RGB")
    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    image = image.resize((target_width, target_height))
    frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    return frame_tensor.unsqueeze(0).unsqueeze(2)

def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:
    """
    Calculate the padding required to resize a source image to a target resolution.

    This function computes the padding needed on each side (top, bottom, left, and right) 
    to resize an image from its source dimensions to the target dimensions. The padding 
    ensures that the resized image maintains its center alignment.

    Args:
        source_height (int): 
            The height of the source image.
        source_width (int): 
            The width of the source image.
        target_height (int): 
            The desired height of the output image.
        target_width (int): 
            The desired width of the output image.

    Returns:
        tuple[int, int, int, int]: 
            A tuple containing the padding values in the format `(pad_left, pad_right, pad_top, pad_bottom)`.

    Example:
        >>> calculate_padding(480, 640, 512, 768)
        (64, 64, 16, 16)
    """
    pad_height = target_height - source_height
    pad_width = target_width - source_width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)

def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    """
    Convert a text prompt into a sanitized filename-friendly string.

    This function processes the input text to:
    1. Remove non-alphabetic characters while retaining spaces.
    2. Convert all characters to lowercase.
    3. Truncate the text to the specified maximum length in words.
    4. Replace spaces with hyphens to create a filename-friendly string.

    Args:
        text (str): 
            The input text prompt to be converted into a filename.
        max_len (int, optional): 
            The maximum length of the resulting filename in terms of character count, excluding hyphens. Defaults to 20.

    Returns:
        str: 
            A sanitized string derived from the input text, suitable for use as part of a filename.

    Example:
        >>> convert_prompt_to_filename("Hello World! This is an example prompt.", max_len=15)
        'hello-world-this'
    """
    clean_text = "".join(char.lower() for char in text if char.isalpha() or char.isspace())
    words = clean_text.split()
    result = []
    current_length = 0
    for word in words:
        new_length = current_length + len(word)
        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break
    return "-".join(result)

def get_unique_filename(
    base: str,
    ext: str,
    prompt: str,
    seed: int,
    resolution: tuple[int, int, int],
    dir: Path,
    endswith: Optional[str] = None,
    index_range: int = 1000
) -> Path:
    """
    Generate a unique filename by appending an incremental index to the base name.

    This function creates a filename based on the provided base name, prompt, seed, resolution, 
    and directory. It ensures uniqueness by appending an incremental index if a file with the 
    generated name already exists. If a unique name cannot be found within the specified range, 
    it raises an exception.

    Args:
        base (str): 
            The base name of the file.
        ext (str): 
            The file extension (e.g., `.png`, `.mp4`).
        prompt (str): 
            A text prompt to include in the filename.
        seed (int): 
            The random seed used in generation, included in the filename for reproducibility.
        resolution (tuple[int, int, int]): 
            A tuple representing the resolution of the output (height, width, frames).
        dir (Path): 
            The directory where the file will be saved.
        endswith (Optional[str], optional): 
            An optional string to append to the filename (e.g., "_condition"). Defaults to None.
        index_range (int, optional): 
            The maximum number of attempts to find a unique filename. Defaults to 1000.

    Returns:
        Path: 
            A unique file path constructed based on the input parameters.

    Raises:
        FileExistsError: 
            If a unique filename cannot be generated within the specified index range.

    Example:
        >>> from pathlib import Path
        >>> filename = get_unique_filename(
        ...     base="output", ext=".mp4", prompt="Sample Prompt", seed=42, 
        ...     resolution=(512, 768, 121), dir=Path("outputs"), endswith="_v1"
        ... )
        >>> print(filename)
        outputs/output_sample-prompt_42_512x768x121_0_v1.mp4
    """
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    for i in range(index_range):
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(f"Could not find a unique filename after {index_range} attempts.")

def seed_everything(seed: int) -> None:
    """
    Set the random seed for reproducibility across different libraries.

    This function ensures that experiments are deterministic by seeding the 
    random number generators of Python's `random`, NumPy, and PyTorch. 
    If a CUDA device is available, the seed is also set for PyTorch's CUDA backend.

    Args:
        seed (int): 
            The seed value to use for random number generation.

    Returns:
        None

    Example:
        >>> seed_everything(42)
        # All random number generators are now seeded with 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def run_pipeline(
    ckpt_dir: Union[str, Path],
    input_image_path: Optional[Union[str, Path]] = None,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    height: int = 480,
    width: int = 704,
    num_frames: int = 121,
    frame_rate: int = 25,
    num_inference_steps: int = 40,
    num_images_per_prompt: int = 1,
    guidance_scale: float = 3.0,
    seed: int = 171198,
    output_path: Optional[Union[str, Path]] = None,
    bfloat16: bool = False
) -> None:
    """
    Run the video generation pipeline and save the results.

    This function initializes and runs the LTX video generation pipeline based on
    the provided parameters. It supports loading pre-trained models and saving
    generated videos or images.

    Args:
        ckpt_dir (Union[str, Path]): 
            Path to the directory containing model checkpoints for VAE, U-Net, and scheduler.
        input_image_path (Optional[Union[str, Path]], optional): 
            Path to the input image file for conditional generation. Defaults to None.
        prompt (Optional[str], optional): 
            Text prompt to guide the video generation. Defaults to None.
        negative_prompt (Optional[str], optional): 
            Negative prompt to avoid undesired features in the generation. Defaults to None.
        height (int, optional): 
            Height of the output video frames. Defaults to 480.
        width (int, optional): 
            Width of the output video frames. Defaults to 704.
        num_frames (int, optional): 
            Number of frames in the generated video. Defaults to 121.
        frame_rate (int, optional): 
            Frame rate for the output video. Defaults to 25.
        num_inference_steps (int, optional): 
            Number of inference steps for the pipeline. Defaults to 40.
        num_images_per_prompt (int, optional): 
            Number of videos or images generated per prompt. Defaults to 1.
        guidance_scale (float, optional): 
            Guidance scale for controlling prompt adherence. Defaults to 3.0.
        seed (int, optional): 
            Seed value for reproducibility. Defaults to 171198.
        output_path (Optional[Union[str, Path]], optional): 
            Directory to save the generated outputs. Defaults to None.
        bfloat16 (bool, optional): 
            Whether to use bfloat16 precision for processing. Defaults to False.

    Returns:
        None: 
            The function saves the outputs to the specified directory and does not return any value.

    Raises:
        FileNotFoundError: 
            If any required model files or input paths are missing.
        RuntimeError: 
            If the pipeline fails during execution.

    Example:
        >>> run_pipeline(
        ...     ckpt_dir="models/checkpoints",
        ...     prompt="A serene mountain landscape",
        ...     height=512, width=768, num_frames=150
        ... )
    """
    global g_vae, g_unet, g_scheduler, g_patchifier, g_text_encoder, g_tokenizer, g_pipeline

    logger = logging.get_logger(__name__)
    logger.warning("Running pipeline with provided parameters.")

    seed_everything(seed)

    output_dir = Path(output_path) if output_path else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_image_path:
        media_items_prepad = load_image_to_tensor_with_resize_and_crop(input_image_path, height, width)
    else:
        media_items_prepad = None

    height_padded = ((height - 1) // 32 + 1) * 32
    width_padded = ((width - 1) // 32 + 1) * 32
    num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1
    padding = calculate_padding(height, width, height_padded, width_padded)

    if media_items_prepad is not None:
        media_items = F.pad(media_items_prepad, padding, mode="constant", value=-1)
    else:
        media_items = None

    ckpt_dir = Path(ckpt_dir)

    if g_vae is None:
        g_vae = load_vae(ckpt_dir / "vae")

    if g_unet is None:
        g_unet = load_unet(ckpt_dir / "unet")
        if bfloat16 and g_unet.dtype != torch.bfloat16:
            g_unet = g_unet.to(torch.bfloat16)

    if g_scheduler is None:
        g_scheduler = load_scheduler(ckpt_dir / "scheduler")

    if g_patchifier is None:
        g_patchifier = SymmetricPatchifier(patch_size=1)

    if g_text_encoder is None:
        g_text_encoder = T5EncoderModel.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder")
        if torch.cuda.is_available():
            g_text_encoder = g_text_encoder.to("cuda")

    if g_tokenizer is None:
        g_tokenizer = T5Tokenizer.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer")

    if g_pipeline is None:
        g_pipeline = LTXVideoPipeline(
            transformer=g_unet,
            patchifier=g_patchifier,
            text_encoder=g_text_encoder,
            tokenizer=g_tokenizer,
            scheduler=g_scheduler,
            vae=g_vae
        )
        if torch.cuda.is_available():
            g_pipeline = g_pipeline.to("cuda")

    sample = {
        "prompt": prompt,
        "prompt_attention_mask": None,
        "negative_prompt": negative_prompt,
        "negative_prompt_attention_mask": None,
        "media_items": media_items,
    }

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

    images = g_pipeline(
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
        height=height_padded,
        width=width_padded,
        num_frames=num_frames_padded,
        frame_rate=frame_rate,
        **sample,
        is_video=True,
        vae_per_channel_normalize=True,
        conditioning_method=(ConditioningMethod.FIRST_FRAME if media_items is not None else ConditioningMethod.UNCONDITIONAL),
        mixed_precision=not bfloat16,
    ).images

    # Crop the padded images to the desired resolution and number of frames
    (pad_left, pad_right, pad_top, pad_bottom) = padding
    pad_bottom = -pad_bottom
    pad_right = -pad_right
    if pad_bottom == 0:
        pad_bottom = images.shape[3]
    if pad_right == 0:
        pad_right = images.shape[4]
    images = images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]

    for i in range(images.shape[0]):
        # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
        video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
        # Unnormalizing images to [0, 255] range
        video_np = (video_np * 255).astype(np.uint8)
        fps = frame_rate
        height, width = video_np.shape[1:3]
        # In case a single image is generated
        if video_np.shape[0] == 1:
            output_filename = get_unique_filename(
                f"image_output_{i}",
                ".png",
                prompt=prompt,
                seed=seed,
                resolution=(height, width, num_frames),
                dir=output_dir,
            )
            imageio.imwrite(output_filename, video_np[0])
        else:
            if input_image_path:
                base_filename = f"img_to_vid_{i}"
            else:
                base_filename = f"text_to_vid_{i}"
            output_filename = get_unique_filename(
                base_filename,
                ".mp4",
                prompt=prompt,
                seed=seed,
                resolution=(height, width, num_frames),
                dir=output_dir,
            )

            # Write video
            with imageio.get_writer(output_filename, fps=fps) as video:
                for frame in video_np:
                    video.append_data(frame)

            # Write condition image
            if input_image_path:
                reference_image = (
                    (
                        media_items_prepad[0, :, 0].permute(1, 2, 0).cpu().data.numpy()
                        + 1.0
                    )
                    / 2.0
                    * 255
                )
                imageio.imwrite(
                    get_unique_filename(
                        base_filename,
                        ".png",
                        prompt=prompt,
                        seed=seed,
                        resolution=(height, width, num_frames),
                        dir=output_dir,
                        endswith="_condition",
                    ),
                    reference_image.astype(np.uint8),
                )

    logger.warning(f"Output saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline from the command line.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to the directory containing model checkpoints.")
    parser.add_argument("--input_image_path", type=str, help="Path to the input image file.")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation.")
    parser.add_argument("--negative_prompt", type=str, default="worst quality, blurry, jittery, distorted",
                        help="Negative prompt for undesired features.")
    parser.add_argument("--height", type=int, default=480, help="Height of the output video frames.")
    parser.add_argument("--width", type=int, default=704, help="Width of the output video frames.")
    parser.add_argument("--num_frames", type=int, default=121, help="Number of frames to generate.")
    parser.add_argument("--frame_rate", type=int, default=25, help="Frame rate for the output video.")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Number of inference steps.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images per prompt.")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale for generation.")
    parser.add_argument("--seed", type=int, default=171198, help="Random seed for reproducibility.")
    parser.add_argument("--output_path", type=str, default=None, help="Output directory for generated content.")
    parser.add_argument("--bfloat16", action="store_true", help="Enable bfloat16 precision.")
    args = parser.parse_args()

    run_pipeline(**vars(args))
