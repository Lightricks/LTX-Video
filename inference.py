import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging

import imageio
import numpy as np
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
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257


def get_total_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return total_memory
    return None


def load_image_to_tensor_with_resize_and_crop(
    image_path, target_height=512, target_width=768
):
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
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)


def calculate_padding(
    source_height: int, source_width: int, target_height: int, target_width: int
) -> tuple[int, int, int, int]:

    # Calculate total padding needed
    pad_height = target_height - source_height
    pad_width = target_width - source_width

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top  # Handles odd padding
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left  # Handles odd padding

    # Return padded tensor
    # Padding format is (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return padding


def convert_prompt_to_filename(text: str, max_len: int = 20) -> str:
    # Remove non-letters and convert to lowercase
    clean_text = "".join(
        char.lower() for char in text if char.isalpha() or char.isspace()
    )

    # Split into words
    words = clean_text.split()

    # Build result string keeping track of length
    result = []
    current_length = 0

    for word in words:
        # Add word length plus 1 for underscore (except for first word)
        new_length = current_length + len(word)

        if new_length <= max_len:
            result.append(word)
            current_length += len(word)
        else:
            break

    return "-".join(result)


# Generate output video name
def get_unique_filename(
    base: str,
    ext: str,
    prompt: str,
    seed: int,
    resolution: tuple[int, int, int],
    dir: Path,
    endswith=None,
    index_range=1000,
) -> Path:
    base_filename = f"{base}_{convert_prompt_to_filename(prompt, max_len=30)}_{seed}_{resolution[0]}x{resolution[1]}x{resolution[2]}"
    for i in range(index_range):
        filename = dir / f"{base_filename}_{i}{endswith if endswith else ''}{ext}"
        if not os.path.exists(filename):
            return filename
    raise FileExistsError(
        f"Could not find a unique filename after {index_range} attempts."
    )


def seed_everething(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Load models from separate directories and run the pipeline."
    )

    # Directories
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to a safetensors file that contains all model parts.",
    )
    parser.add_argument(
        "--input_video_path",
        type=str,
        help="Path to the input video file (first frame used)",
    )
    parser.add_argument(
        "--input_image_path", type=str, help="Path to the input image file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to the folder to save output video, if None will save in outputs/ directory.",
    )
    parser.add_argument("--seed", type=int, default="171198")

    # Pipeline parameters
    parser.add_argument(
        "--num_inference_steps", type=int, default=40, help="Number of inference steps"
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images per prompt",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3,
        help="Guidance scale for the pipeline",
    )
    parser.add_argument(
        "--stg_scale",
        type=float,
        default=1,
        help="Spatiotemporal guidance scale for the pipeline. 0 to disable STG.",
    )
    parser.add_argument(
        "--stg_rescale",
        type=float,
        default=0.7,
        help="Spatiotemporal guidance rescaling scale for the pipeline. 1 to disable rescale.",
    )
    parser.add_argument(
        "--stg_mode",
        type=str,
        default="stg_a",
        help="Spatiotemporal guidance mode for the pipeline. Can be either stg_a or stg_r.",
    )
    parser.add_argument(
        "--stg_skip_layers",
        type=str,
        default="19",
        help="Attention layers to skip for spatiotemporal guidance. Comma separated list of integers.",
    )
    parser.add_argument(
        "--image_cond_noise_scale",
        type=float,
        default=0.15,
        help="Amount of noise to add to the conditioned image",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of the output video frames. Optional if an input image provided.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=704,
        help="Width of the output video frames. If None will infer from input image.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=121,
        help="Number of frames to generate in the output video",
    )
    parser.add_argument(
        "--frame_rate", type=int, default=25, help="Frame rate for the output video"
    )

    parser.add_argument(
        "--precision",
        choices=["bfloat16", "mixed_precision"],
        default="bfloat16",
        help="Sets the precision for the transformer and tokenizer. Default is bfloat16. If 'mixed_precision' is enabled, it moves to mixed-precision.",
    )

    # VAE noise augmentation
    parser.add_argument(
        "--decode_timestep",
        type=float,
        default=0.05,
        help="Timestep for decoding noise",
    )
    parser.add_argument(
        "--decode_noise_scale",
        type=float,
        default=0.025,
        help="Noise level for decoding noise",
    )

    # Prompts
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt to guide generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        help="Negative prompt for undesired features",
    )

    parser.add_argument(
        "--offload_to_cpu",
        action="store_true",
        help="Offloading unnecessary computations to CPU.",
    )

    logger = logging.get_logger(__name__)

    args = parser.parse_args()

    logger.warning(f"Running generation with arguments: {args}")

    seed_everething(args.seed)

    offload_to_cpu = False if not args.offload_to_cpu else get_total_gpu_memory() < 30

    output_dir = (
        Path(args.output_path)
        if args.output_path
        else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    if args.input_image_path:
        media_items_prepad = load_image_to_tensor_with_resize_and_crop(
            args.input_image_path, args.height, args.width
        )
    else:
        media_items_prepad = None

    height = args.height if args.height else media_items_prepad.shape[-2]
    width = args.width if args.width else media_items_prepad.shape[-1]
    num_frames = args.num_frames

    if height > MAX_HEIGHT or width > MAX_WIDTH or num_frames > MAX_NUM_FRAMES:
        logger.warning(
            f"Input resolution or number of frames {height}x{width}x{num_frames} is too big, it is suggested to use the resolution below {MAX_HEIGHT}x{MAX_WIDTH}x{MAX_NUM_FRAMES}."
        )

    # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
    height_padded = ((height - 1) // 32 + 1) * 32
    width_padded = ((width - 1) // 32 + 1) * 32
    num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1

    padding = calculate_padding(height, width, height_padded, width_padded)

    logger.warning(
        f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}"
    )

    if media_items_prepad is not None:
        media_items = F.pad(
            media_items_prepad, padding, mode="constant", value=-1
        )  # -1 is the value for padding since the image is normalized to -1, 1
    else:
        media_items = None

    ckpt_path = Path(args.ckpt_path)
    vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
    transformer = Transformer3DModel.from_pretrained(ckpt_path)
    scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)

    text_encoder = T5EncoderModel.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
    )
    patchifier = SymmetricPatchifier(patch_size=1)
    tokenizer = T5Tokenizer.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
    )

    if torch.cuda.is_available():
        transformer = transformer.cuda()
        vae = vae.cuda()
        text_encoder = text_encoder.cuda()

    vae = vae.to(torch.bfloat16)
    if args.precision == "bfloat16" and transformer.dtype != torch.bfloat16:
        transformer = transformer.to(torch.bfloat16)
    text_encoder = text_encoder.to(torch.bfloat16)

    # Set spatiotemporal guidance
    skip_block_list = [int(x.strip()) for x in args.stg_skip_layers.split(",")]
    skip_layer_strategy = (
        SkipLayerStrategy.Attention
        if args.stg_mode.lower() == "stg_a"
        else SkipLayerStrategy.Residual
    )

    # Use submodels for the pipeline
    submodel_dict = {
        "transformer": transformer,
        "patchifier": patchifier,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "vae": vae,
    }

    pipeline = LTXVideoPipeline(**submodel_dict)
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")

    # Prepare input for the pipeline
    sample = {
        "prompt": args.prompt,
        "prompt_attention_mask": None,
        "negative_prompt": args.negative_prompt,
        "negative_prompt_attention_mask": None,
        "media_items": media_items,
    }

    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(args.seed)

    images = pipeline(
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.num_images_per_prompt,
        guidance_scale=args.guidance_scale,
        skip_layer_strategy=skip_layer_strategy,
        skip_block_list=skip_block_list,
        stg_scale=args.stg_scale,
        do_rescaling=args.stg_rescale != 1,
        rescaling_scale=args.stg_rescale,
        generator=generator,
        output_type="pt",
        callback_on_step_end=None,
        height=height_padded,
        width=width_padded,
        num_frames=num_frames_padded,
        frame_rate=args.frame_rate,
        **sample,
        is_video=True,
        vae_per_channel_normalize=True,
        conditioning_method=(
            ConditioningMethod.FIRST_FRAME
            if media_items is not None
            else ConditioningMethod.UNCONDITIONAL
        ),
        image_cond_noise_scale=args.image_cond_noise_scale,
        decode_timestep=args.decode_timestep,
        decode_noise_scale=args.decode_noise_scale,
        mixed_precision=(args.precision == "mixed_precision"),
        offload_to_cpu=offload_to_cpu,
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
        fps = args.frame_rate
        height, width = video_np.shape[1:3]
        # In case a single image is generated
        if video_np.shape[0] == 1:
            output_filename = get_unique_filename(
                f"image_output_{i}",
                ".png",
                prompt=args.prompt,
                seed=args.seed,
                resolution=(height, width, num_frames),
                dir=output_dir,
            )
            imageio.imwrite(output_filename, video_np[0])
        else:
            if args.input_image_path:
                base_filename = f"img_to_vid_{i}"
            else:
                base_filename = f"text_to_vid_{i}"
            output_filename = get_unique_filename(
                base_filename,
                ".mp4",
                prompt=args.prompt,
                seed=args.seed,
                resolution=(height, width, num_frames),
                dir=output_dir,
            )

            # Write video
            with imageio.get_writer(output_filename, fps=fps) as video:
                for frame in video_np:
                    video.append_data(frame)

            # Write condition image
            if args.input_image_path:
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
                        prompt=args.prompt,
                        seed=args.seed,
                        resolution=(height, width, num_frames),
                        dir=output_dir,
                        endswith="_condition",
                    ),
                    reference_image.astype(np.uint8),
                )
        logger.warning(f"Output saved to {output_dir}")


if __name__ == "__main__":
    main()
