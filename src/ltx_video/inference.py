import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from diffusers.utils import logging
from typing import Optional, List, Union
import yaml

import imageio
import json
import numpy as np
import torch
from safetensors import safe_open
from PIL import Image
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from huggingface_hub import hf_hub_download

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.pipelines.pipeline_ltx_video import (
    ConditioningItem,
    LTXVideoPipeline,
    LTXMultiScalePipeline,
)
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler

MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257

logger = logging.get_logger("LTX-Video")


def get_total_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return total_memory
    return 0


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image_to_tensor_with_resize_and_crop(
    image_input: Union[str, Image.Image],
    target_height: int = 512,
    target_width: int = 768,
    just_crop: bool = False,
) -> torch.Tensor:
    """Load and process an image into a tensor.

    Args:
        image_input: Either a file path (str) or a PIL Image object
        target_height: Desired height of output tensor
        target_width: Desired width of output tensor
        just_crop: If True, only crop the image to the target size without resizing
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("image_input must be either a file path or a PIL Image object")

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
    if not just_crop:
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
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Load models from separate directories and run the pipeline."
    )

    # Directories
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to the folder to save output video, if None will save in outputs/ directory.",
    )
    parser.add_argument("--seed", type=int, default="171198")

    # Pipeline parameters
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images per prompt",
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
        default=704,
        help="Height of the output video frames. Optional if an input image provided.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1216,
        help="Width of the output video frames. If None will infer from input image.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=121,
        help="Number of frames to generate in the output video",
    )
    parser.add_argument(
        "--frame_rate", type=int, default=30, help="Frame rate for the output video"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run inference on. If not specified, will automatically detect and use CUDA or MPS if available, else CPU.",
    )
    parser.add_argument(
        "--pipeline_config",
        type=str,
        default="configs/ltxv-13b-0.9.7-dev.yaml",
        help="The path to the config file for the pipeline, which contains the parameters for the pipeline",
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

    # video-to-video arguments:
    parser.add_argument(
        "--input_media_path",
        type=str,
        default=None,
        help="Path to the input video (or imaage) to be modified using the video-to-video pipeline",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Editing strength (noising level) for video-to-video pipeline.",
    )

    # Conditioning arguments
    parser.add_argument(
        "--conditioning_media_paths",
        type=str,
        nargs="*",
        help="List of paths to conditioning media (images or videos). Each path will be used as a conditioning item.",
    )
    parser.add_argument(
        "--conditioning_strengths",
        type=float,
        nargs="*",
        help="List of conditioning strengths (between 0 and 1) for each conditioning item. Must match the number of conditioning items.",
    )
    parser.add_argument(
        "--conditioning_start_frames",
        type=int,
        nargs="*",
        help="List of frame indices where each conditioning item should be applied. Must match the number of conditioning items.",
    )

    args = parser.parse_args()
    logger.warning(f"Running generation with arguments: {args}")
    infer(**vars(args))


def create_ltx_video_pipeline(
    ckpt_path: str,
    precision: str,
    text_encoder_model_name_or_path: str,
    sampler: Optional[str] = None,
    device: Optional[str] = None,
    enhance_prompt: bool = False,
    prompt_enhancer_image_caption_model_name_or_path: Optional[str] = None,
    prompt_enhancer_llm_model_name_or_path: Optional[str] = None,
) -> LTXVideoPipeline:
    ckpt_path = Path(ckpt_path)
    assert os.path.exists(
        ckpt_path
    ), f"Ckpt path provided (--ckpt_path) {ckpt_path} does not exist"

    with safe_open(ckpt_path, framework="pt") as f:
        metadata = f.metadata()
        config_str = metadata.get("config")
        configs = json.loads(config_str)
        allowed_inference_steps = configs.get("allowed_inference_steps", None)

    vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
    transformer = Transformer3DModel.from_pretrained(ckpt_path)

    # Use constructor if sampler is specified, otherwise use from_pretrained
    if sampler == "from_checkpoint" or not sampler:
        scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)
    else:
        scheduler = RectifiedFlowScheduler(
            sampler=("Uniform" if sampler.lower() == "uniform" else "LinearQuadratic")
        )

    text_encoder = T5EncoderModel.from_pretrained(
        text_encoder_model_name_or_path, subfolder="text_encoder"
    )
    patchifier = SymmetricPatchifier(patch_size=1)
    tokenizer = T5Tokenizer.from_pretrained(
        text_encoder_model_name_or_path, subfolder="tokenizer"
    )

    transformer = transformer.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    if enhance_prompt:
        prompt_enhancer_image_caption_model = AutoModelForCausalLM.from_pretrained(
            prompt_enhancer_image_caption_model_name_or_path, trust_remote_code=True
        )
        prompt_enhancer_image_caption_processor = AutoProcessor.from_pretrained(
            prompt_enhancer_image_caption_model_name_or_path, trust_remote_code=True
        )
        prompt_enhancer_llm_model = AutoModelForCausalLM.from_pretrained(
            prompt_enhancer_llm_model_name_or_path,
            torch_dtype="bfloat16",
        )
        prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained(
            prompt_enhancer_llm_model_name_or_path,
        )
    else:
        prompt_enhancer_image_caption_model = None
        prompt_enhancer_image_caption_processor = None
        prompt_enhancer_llm_model = None
        prompt_enhancer_llm_tokenizer = None

    vae = vae.to(torch.bfloat16)
    if precision == "bfloat16" and transformer.dtype != torch.bfloat16:
        transformer = transformer.to(torch.bfloat16)
    text_encoder = text_encoder.to(torch.bfloat16)

    # Use submodels for the pipeline
    submodel_dict = {
        "transformer": transformer,
        "patchifier": patchifier,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "vae": vae,
        "prompt_enhancer_image_caption_model": prompt_enhancer_image_caption_model,
        "prompt_enhancer_image_caption_processor": prompt_enhancer_image_caption_processor,
        "prompt_enhancer_llm_model": prompt_enhancer_llm_model,
        "prompt_enhancer_llm_tokenizer": prompt_enhancer_llm_tokenizer,
        "allowed_inference_steps": allowed_inference_steps,
    }

    pipeline = LTXVideoPipeline(**submodel_dict)
    pipeline = pipeline.to(device)
    return pipeline


def create_latent_upsampler(latent_upsampler_model_path: str, device: str):
    latent_upsampler = LatentUpsampler.from_pretrained(latent_upsampler_model_path)
    latent_upsampler.to(device)
    latent_upsampler.eval()
    return latent_upsampler


def infer(
    output_path: Optional[str],
    seed: int,
    pipeline_config: str,
    image_cond_noise_scale: float,
    height: Optional[int],
    width: Optional[int],
    num_frames: int,
    frame_rate: int,
    prompt: str,
    negative_prompt: str,
    offload_to_cpu: bool,
    input_media_path: Optional[str] = None,
    strength: Optional[float] = 1.0,
    conditioning_media_paths: Optional[List[str]] = None,
    conditioning_strengths: Optional[List[float]] = None,
    conditioning_start_frames: Optional[List[int]] = None,
    device: Optional[str] = None,
    **kwargs,
):
    # check if pipeline_config is a file
    if not os.path.isfile(pipeline_config):
        raise ValueError(f"Pipeline config file {pipeline_config} does not exist")
    with open(pipeline_config, "r") as f:
        pipeline_config = yaml.safe_load(f)

    models_dir = "MODEL_DIR"

    ltxv_model_name_or_path = pipeline_config["checkpoint_path"]
    if not os.path.isfile(ltxv_model_name_or_path):
        ltxv_model_path = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=ltxv_model_name_or_path,
            local_dir=models_dir,
            repo_type="model",
        )
    else:
        ltxv_model_path = ltxv_model_name_or_path

    spatial_upscaler_model_name_or_path = pipeline_config.get(
        "spatial_upscaler_model_path"
    )
    if spatial_upscaler_model_name_or_path and not os.path.isfile(
        spatial_upscaler_model_name_or_path
    ):
        spatial_upscaler_model_path = hf_hub_download(
            repo_id="Lightricks/LTX-Video",
            filename=spatial_upscaler_model_name_or_path,
            local_dir=models_dir,
            repo_type="model",
        )
    else:
        spatial_upscaler_model_path = spatial_upscaler_model_name_or_path

    if kwargs.get("input_image_path", None):
        logger.warning(
            "Please use conditioning_media_paths instead of input_image_path."
        )
        assert not conditioning_media_paths and not conditioning_start_frames
        conditioning_media_paths = [kwargs["input_image_path"]]
        conditioning_start_frames = [0]

    # Validate conditioning arguments
    if conditioning_media_paths:
        # Use default strengths of 1.0
        if not conditioning_strengths:
            conditioning_strengths = [1.0] * len(conditioning_media_paths)
        if not conditioning_start_frames:
            raise ValueError(
                "If `conditioning_media_paths` is provided, "
                "`conditioning_start_frames` must also be provided"
            )
        if len(conditioning_media_paths) != len(conditioning_strengths) or len(
            conditioning_media_paths
        ) != len(conditioning_start_frames):
            raise ValueError(
                "`conditioning_media_paths`, `conditioning_strengths`, "
                "and `conditioning_start_frames` must have the same length"
            )
        if any(s < 0 or s > 1 for s in conditioning_strengths):
            raise ValueError("All conditioning strengths must be between 0 and 1")
        if any(f < 0 or f >= num_frames for f in conditioning_start_frames):
            raise ValueError(
                f"All conditioning start frames must be between 0 and {num_frames-1}"
            )

    seed_everething(seed)
    if offload_to_cpu and not torch.cuda.is_available():
        logger.warning(
            "offload_to_cpu is set to True, but offloading will not occur since the model is already running on CPU."
        )
        offload_to_cpu = False
    else:
        offload_to_cpu = offload_to_cpu and get_total_gpu_memory() < 30

    output_dir = (
        Path(output_path)
        if output_path
        else Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Adjust dimensions to be divisible by 32 and num_frames to be (N * 8 + 1)
    height_padded = ((height - 1) // 32 + 1) * 32
    width_padded = ((width - 1) // 32 + 1) * 32
    num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1

    padding = calculate_padding(height, width, height_padded, width_padded)

    logger.warning(
        f"Padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}"
    )

    prompt_enhancement_words_threshold = pipeline_config[
        "prompt_enhancement_words_threshold"
    ]

    prompt_word_count = len(prompt.split())
    enhance_prompt = (
        prompt_enhancement_words_threshold > 0
        and prompt_word_count < prompt_enhancement_words_threshold
    )

    if prompt_enhancement_words_threshold > 0 and not enhance_prompt:
        logger.info(
            f"Prompt has {prompt_word_count} words, which exceeds the threshold of {prompt_enhancement_words_threshold}. Prompt enhancement disabled."
        )

    precision = pipeline_config["precision"]
    text_encoder_model_name_or_path = pipeline_config["text_encoder_model_name_or_path"]
    sampler = pipeline_config["sampler"]
    prompt_enhancer_image_caption_model_name_or_path = pipeline_config[
        "prompt_enhancer_image_caption_model_name_or_path"
    ]
    prompt_enhancer_llm_model_name_or_path = pipeline_config[
        "prompt_enhancer_llm_model_name_or_path"
    ]

    pipeline = create_ltx_video_pipeline(
        ckpt_path=ltxv_model_path,
        precision=precision,
        text_encoder_model_name_or_path=text_encoder_model_name_or_path,
        sampler=sampler,
        device=kwargs.get("device", get_device()),
        enhance_prompt=enhance_prompt,
        prompt_enhancer_image_caption_model_name_or_path=prompt_enhancer_image_caption_model_name_or_path,
        prompt_enhancer_llm_model_name_or_path=prompt_enhancer_llm_model_name_or_path,
    )

    if pipeline_config.get("pipeline_type", None) == "multi-scale":
        if not spatial_upscaler_model_path:
            raise ValueError(
                "spatial upscaler model path is missing from pipeline config file and is required for multi-scale rendering"
            )
        latent_upsampler = create_latent_upsampler(
            spatial_upscaler_model_path, pipeline.device
        )
        pipeline = LTXMultiScalePipeline(pipeline, latent_upsampler=latent_upsampler)

    media_item = None
    if input_media_path:
        media_item = load_media_file(
            media_path=input_media_path,
            height=height,
            width=width,
            max_frames=num_frames_padded,
            padding=padding,
        )

    conditioning_items = (
        prepare_conditioning(
            conditioning_media_paths=conditioning_media_paths,
            conditioning_strengths=conditioning_strengths,
            conditioning_start_frames=conditioning_start_frames,
            height=height,
            width=width,
            num_frames=num_frames,
            padding=padding,
            pipeline=pipeline,
        )
        if conditioning_media_paths
        else None
    )

    stg_mode = pipeline_config.get("stg_mode", "attention_values")
    del pipeline_config["stg_mode"]
    if stg_mode.lower() == "stg_av" or stg_mode.lower() == "attention_values":
        skip_layer_strategy = SkipLayerStrategy.AttentionValues
    elif stg_mode.lower() == "stg_as" or stg_mode.lower() == "attention_skip":
        skip_layer_strategy = SkipLayerStrategy.AttentionSkip
    elif stg_mode.lower() == "stg_r" or stg_mode.lower() == "residual":
        skip_layer_strategy = SkipLayerStrategy.Residual
    elif stg_mode.lower() == "stg_t" or stg_mode.lower() == "transformer_block":
        skip_layer_strategy = SkipLayerStrategy.TransformerBlock
    else:
        raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")

    # Prepare input for the pipeline
    sample = {
        "prompt": prompt,
        "prompt_attention_mask": None,
        "negative_prompt": negative_prompt,
        "negative_prompt_attention_mask": None,
    }

    device = device or get_device()
    generator = torch.Generator(device=device).manual_seed(seed)

    images = pipeline(
        **pipeline_config,
        skip_layer_strategy=skip_layer_strategy,
        generator=generator,
        output_type="pt",
        callback_on_step_end=None,
        height=height_padded,
        width=width_padded,
        num_frames=num_frames_padded,
        frame_rate=frame_rate,
        **sample,
        media_items=media_item,
        strength=strength,
        conditioning_items=conditioning_items,
        is_video=True,
        vae_per_channel_normalize=True,
        image_cond_noise_scale=image_cond_noise_scale,
        mixed_precision=(precision == "mixed_precision"),
        offload_to_cpu=offload_to_cpu,
        device=device,
        enhance_prompt=enhance_prompt,
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
            output_filename = get_unique_filename(
                f"video_output_{i}",
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

        logger.warning(f"Output saved to {output_filename}")


def prepare_conditioning(
    conditioning_media_paths: List[str],
    conditioning_strengths: List[float],
    conditioning_start_frames: List[int],
    height: int,
    width: int,
    num_frames: int,
    padding: tuple[int, int, int, int],
    pipeline: LTXVideoPipeline,
) -> Optional[List[ConditioningItem]]:
    """Prepare conditioning items based on input media paths and their parameters.

    Args:
        conditioning_media_paths: List of paths to conditioning media (images or videos)
        conditioning_strengths: List of conditioning strengths for each media item
        conditioning_start_frames: List of frame indices where each item should be applied
        height: Height of the output frames
        width: Width of the output frames
        num_frames: Number of frames in the output video
        padding: Padding to apply to the frames
        pipeline: LTXVideoPipeline object used for condition video trimming

    Returns:
        A list of ConditioningItem objects.
    """
    conditioning_items = []
    for path, strength, start_frame in zip(
        conditioning_media_paths, conditioning_strengths, conditioning_start_frames
    ):
        num_input_frames = orig_num_input_frames = get_media_num_frames(path)
        if hasattr(pipeline, "trim_conditioning_sequence") and callable(
            getattr(pipeline, "trim_conditioning_sequence")
        ):
            num_input_frames = pipeline.trim_conditioning_sequence(
                start_frame, orig_num_input_frames, num_frames
            )
        if num_input_frames < orig_num_input_frames:
            logger.warning(
                f"Trimming conditioning video {path} from {orig_num_input_frames} to {num_input_frames} frames."
            )

        media_tensor = load_media_file(
            media_path=path,
            height=height,
            width=width,
            max_frames=num_input_frames,
            padding=padding,
            just_crop=True,
        )
        conditioning_items.append(ConditioningItem(media_tensor, start_frame, strength))
    return conditioning_items


def get_media_num_frames(media_path: str) -> int:
    is_video = any(
        media_path.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]
    )
    num_frames = 1
    if is_video:
        reader = imageio.get_reader(media_path)
        num_frames = reader.count_frames()
        reader.close()
    return num_frames


def load_media_file(
    media_path: str,
    height: int,
    width: int,
    max_frames: int,
    padding: tuple[int, int, int, int],
    just_crop: bool = False,
) -> torch.Tensor:
    is_video = any(
        media_path.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]
    )
    if is_video:
        reader = imageio.get_reader(media_path)
        num_input_frames = min(reader.count_frames(), max_frames)

        # Read and preprocess the relevant frames from the video file.
        frames = []
        for i in range(num_input_frames):
            frame = Image.fromarray(reader.get_data(i))
            frame_tensor = load_image_to_tensor_with_resize_and_crop(
                frame, height, width, just_crop=just_crop
            )
            frame_tensor = torch.nn.functional.pad(frame_tensor, padding)
            frames.append(frame_tensor)
        reader.close()

        # Stack frames along the temporal dimension
        media_tensor = torch.cat(frames, dim=2)
    else:  # Input image
        media_tensor = load_image_to_tensor_with_resize_and_crop(
            media_path, height, width, just_crop=just_crop
        )
        media_tensor = torch.nn.functional.pad(media_tensor, padding)
    return media_tensor


if __name__ == "__main__":
    main()
