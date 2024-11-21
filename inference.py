import torch
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from pathlib import Path
from transformers import T5EncoderModel, T5Tokenizer
import safetensors.torch
import json
import argparse
from ltx_video.utils.conditioning_method import ConditioningMethod
import os
import numpy as np
import cv2
from PIL import Image
import random

RECOMMENDED_RESOLUTIONS = [
    (704, 1216, 41),
    (704, 1088, 49),
    (640, 1056, 57),
    (608, 992, 65),
    (608, 896, 73),
    (544, 896, 81),
    (544, 832, 89),
    (512, 800, 97),
    (512, 768, 97),
    (480, 800, 105),
    (480, 736, 113),
    (480, 704, 121),
    (448, 704, 129),
    (448, 672, 137),
    (416, 640, 153),
    (384, 672, 161),
    (384, 640, 169),
    (384, 608, 177),
    (384, 576, 185),
    (352, 608, 193),
    (352, 576, 201),
    (352, 544, 209),
    (352, 512, 225),
    (352, 512, 233),
    (320, 544, 241),
    (320, 512, 249),
    (320, 512, 257),
]


def load_vae(vae_dir):
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


def load_unet(unet_dir):
    unet_ckpt_path = unet_dir / "unet_diffusion_pytorch_model.safetensors"
    unet_config_path = unet_dir / "config.json"
    transformer_config = Transformer3DModel.load_config(unet_config_path)
    transformer = Transformer3DModel.from_config(transformer_config)
    unet_state_dict = safetensors.torch.load_file(unet_ckpt_path)
    transformer.load_state_dict(unet_state_dict, strict=True)
    if torch.cuda.is_available():
        transformer = transformer.cuda()
    return transformer


def load_scheduler(scheduler_dir):
    scheduler_config_path = scheduler_dir / "scheduler_config.json"
    scheduler_config = RectifiedFlowScheduler.load_config(scheduler_config_path)
    return RectifiedFlowScheduler.from_config(scheduler_config)


def center_crop_and_resize(frame, target_height, target_width):
    h, w, _ = frame.shape
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = w / h
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(h * aspect_ratio_target)
        x_start = (w - new_width) // 2
        frame_cropped = frame[:, x_start : x_start + new_width]
    else:
        new_height = int(w / aspect_ratio_target)
        y_start = (h - new_height) // 2
        frame_cropped = frame[y_start : y_start + new_height, :]
    frame_resized = cv2.resize(frame_cropped, (target_width, target_height))
    return frame_resized


def load_video_to_tensor_with_resize(video_path, target_height, target_width):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_height is not None:
            frame_resized = center_crop_and_resize(
                frame_rgb, target_height, target_width
            )
        else:
            frame_resized = frame_rgb
        frames.append(frame_resized)
    cap.release()
    video_np = (np.array(frames) / 127.5) - 1.0
    video_tensor = torch.tensor(video_np).permute(3, 0, 1, 2).float()
    return video_tensor


def load_image_to_tensor_with_resize(image_path, target_height=512, target_width=768):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    frame_resized = center_crop_and_resize(image_np, target_height, target_width)
    frame_tensor = torch.tensor(frame_resized).permute(2, 0, 1).float()
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)


def main():
    parser = argparse.ArgumentParser(
        description="Load models from separate directories and run the pipeline."
    )

    # Directories
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to the directory containing unet, vae, and scheduler subdirectories",
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
        help="Path to save output video, if None will save in working directory.",
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
        "--height",
        type=int,
        default=None,
        help="Height of the output video frames. Optional if an input image provided.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
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
        "--bfloat16",
        action="store_true",
        help="Denoise in bfloat16",
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
        "--custom_resolution",
        action="store_true",
        default=False,
        help="Enable custom resolution (not in recommneded resolutions) if specified (default: False)",
    )

    args = parser.parse_args()

    if args.input_image_path is None and args.input_video_path is None:
        assert (
            args.height is not None and args.width is not None
        ), "Must enter height and width for text to image generation."

    # Load media (video or image)
    if args.input_video_path:
        media_items = load_video_to_tensor_with_resize(
            args.input_video_path, args.height, args.width
        ).unsqueeze(0)
    elif args.input_image_path:
        media_items = load_image_to_tensor_with_resize(
            args.input_image_path, args.height, args.width
        )
    else:
        media_items = None

    height = args.height if args.height else media_items.shape[-2]
    width = args.width if args.width else media_items.shape[-1]
    assert height % 32 == 0, f"Height ({height}) should be divisible by 32."
    assert width % 32 == 0, f"Width ({width}) should be divisible by 32."
    assert (
        height,
        width,
        args.num_frames,
    ) in RECOMMENDED_RESOLUTIONS or args.custom_resolution, f"The selected resolution + num frames combination is not supported, results would be suboptimal. Supported (h,w,f) are: {RECOMMENDED_RESOLUTIONS}. Use --custom_resolution to enable working with this resolution."

    # Paths for the separate mode directories
    ckpt_dir = Path(args.ckpt_dir)
    unet_dir = ckpt_dir / "unet"
    vae_dir = ckpt_dir / "vae"
    scheduler_dir = ckpt_dir / "scheduler"

    # Load models
    vae = load_vae(vae_dir)
    unet = load_unet(unet_dir)
    scheduler = load_scheduler(scheduler_dir)
    patchifier = SymmetricPatchifier(patch_size=1)
    text_encoder = T5EncoderModel.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="text_encoder"
    )
    if torch.cuda.is_available():
        text_encoder = text_encoder.to("cuda")
    tokenizer = T5Tokenizer.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", subfolder="tokenizer"
    )

    if args.bfloat16 and unet.dtype != torch.bfloat16:
        unet = unet.to(torch.bfloat16)

    # Use submodels for the pipeline
    submodel_dict = {
        "transformer": unet,
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(args.seed)

    images = pipeline(
        num_inference_steps=args.num_inference_steps,
        num_images_per_prompt=args.num_images_per_prompt,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="pt",
        callback_on_step_end=None,
        height=height,
        width=width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        **sample,
        is_video=True,
        vae_per_channel_normalize=True,
        conditioning_method=(
            ConditioningMethod.FIRST_FRAME
            if media_items is not None
            else ConditioningMethod.UNCONDITIONAL
        ),
        mixed_precision=not args.bfloat16,
    ).images

    # Save output video
    def get_unique_filename(base, ext, dir=".", index_range=1000):
        for i in range(index_range):
            filename = os.path.join(dir, f"{base}_{i}{ext}")
            if not os.path.exists(filename):
                return filename
        raise FileExistsError(
            f"Could not find a unique filename after {index_range} attempts."
        )

    for i in range(images.shape[0]):
        # Gathering from B, C, F, H, W to C, F, H, W and then permuting to F, H, W, C
        video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
        # Unnormalizing images to [0, 255] range
        video_np = (video_np * 255).astype(np.uint8)
        fps = args.frame_rate
        height, width = video_np.shape[1:3]
        if video_np.shape[0] == 1:
            output_filename = (
                args.output_path
                if args.output_path is not None
                else get_unique_filename(f"image_output_{i}", ".png", ".")
            )
            cv2.imwrite(
                output_filename, video_np[0][..., ::-1]
            )  # Save single frame as image
        else:
            output_filename = (
                args.output_path
                if args.output_path is not None
                else get_unique_filename(f"video_output_{i}", ".mp4", ".")
            )

            out = cv2.VideoWriter(
                output_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

            for frame in video_np[..., ::-1]:
                out.write(frame)
            out.release()


if __name__ == "__main__":
    main()
