import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import torch
from huggingface_hub import snapshot_download
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from inference import (
    load_vae, 
    load_unet, 
    load_scheduler,
    load_image_to_tensor_with_resize_and_crop,
    calculate_padding,
    get_unique_filename,
    seed_everething,
    SymmetricPatchifier,
    ConditioningMethod
)
from transformers import T5EncoderModel, T5Tokenizer
import imageio
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for loaded models
pipeline = None
generator = None

# Constants from inference.py
MAX_HEIGHT = 720
MAX_WIDTH = 1280
MAX_NUM_FRAMES = 257

# Model precision configuration
USE_BFLOAT16 = os.getenv("USE_BFLOAT16", "true").lower() == "true"

# Download configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "16"))  # Number of parallel workers for downloading

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, generator
    
    logger.info("Starting LTX Video API server...")
    
    try:
        # Get current working directory and create relative paths
        current_dir = Path.cwd()
        ckpt_dir = current_dir / os.getenv("CKPT_DIR", "model").lstrip("./")
        
        # Download model if not exists
        if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
            logger.info(f"Model not found in {ckpt_dir}. Downloading with {MAX_WORKERS} workers...")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_download(
                "Lightricks/LTX-Video",
                local_dir=str(ckpt_dir),
                local_dir_use_symlinks=False,
                repo_type='model',
                max_workers=MAX_WORKERS,  # Enable parallel downloading
                resume_download=True,     # Resume interrupted downloads
                etag_timeout=30          # Increase timeout for better stability
            )
            logger.info("Model download completed successfully")
        
        unet_dir = ckpt_dir / "unet"
        vae_dir = ckpt_dir / "vae"
        scheduler_dir = ckpt_dir / "scheduler"

        logger.info("Loading VAE model...")
        vae = load_vae(vae_dir)  # This will load VAE in bfloat16
        
        logger.info("Loading UNet model...")
        unet = load_unet(unet_dir)
        if USE_BFLOAT16 and unet.dtype != torch.bfloat16:
            logger.info("Converting UNet to bfloat16...")
            unet = unet.to(torch.bfloat16)
        
        logger.info("Loading scheduler...")
        scheduler = load_scheduler(scheduler_dir)
        patchifier = SymmetricPatchifier(patch_size=1)
        
        logger.info("Loading text encoder and tokenizer...")
        text_encoder = T5EncoderModel.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", 
            subfolder="text_encoder"
        )
        if torch.cuda.is_available():
            text_encoder = text_encoder.to("cuda")
            
        tokenizer = T5Tokenizer.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            subfolder="tokenizer"
        )

        # Initialize pipeline
        logger.info("Initializing LTX Video pipeline...")
        pipeline = LTXVideoPipeline(
            transformer=unet,
            patchifier=patchifier,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            vae=vae
        )
        
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
            logger.info("Pipeline moved to CUDA")
            
        # Initialize generator
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Server startup complete!")
        
        yield
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")
    finally:
        # Cleanup resources if needed
        pipeline = None
        generator = None

# Create FastAPI app with lifespan
app = FastAPI(
    title="LTX Video Generation API",
    description="API for generating videos using LTX-Video model",
    version="1.0.0",
    lifespan=lifespan
)

class GenerationParams(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = Field(
        default=os.getenv("DEFAULT_NEGATIVE_PROMPT"),
        description="Negative prompt for undesired features"
    )
    num_inference_steps: Optional[int] = Field(
        default=int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", 40)),
        description="Number of inference steps"
    )
    guidance_scale: Optional[float] = Field(
        default=float(os.getenv("DEFAULT_GUIDANCE_SCALE", 3.0)),
        description="Guidance scale for the pipeline"
    )
    height: Optional[int] = Field(
        default=int(os.getenv("DEFAULT_HEIGHT", 480)),
        description="Height of the output video frames"
    )
    width: Optional[int] = Field(
        default=int(os.getenv("DEFAULT_WIDTH", 704)), 
        description="Width of the output video frames"
    )
    num_frames: Optional[int] = Field(
        default=int(os.getenv("DEFAULT_NUM_FRAMES", 121)),
        description="Number of frames to generate"
    )
    frame_rate: Optional[int] = Field(
        default=int(os.getenv("DEFAULT_FRAME_RATE", 25)),
        description="Frame rate for the output video"
    )
    seed: Optional[int] = Field(
        default=171198,
        description="Random seed for generation"
    )
    use_mixed_precision: Optional[bool] = Field(
        default=not USE_BFLOAT16,
        description="Whether to use mixed precision during inference"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A clear, turquoise river flows through a rocky canyon",
                "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
                "num_inference_steps": 40,
                "guidance_scale": 3.0,
                "height": 480,
                "width": 704,
                "num_frames": 121,
                "frame_rate": 25,
                "seed": 171198,
                "use_mixed_precision": False
            }
        }

def validate_dimensions(height: int, width: int, num_frames: int):
    """Validate input dimensions against maximum allowed values"""
    if height > MAX_HEIGHT or width > MAX_WIDTH or num_frames > MAX_NUM_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Input dimensions {height}x{width}x{num_frames} exceed maximum allowed values ({MAX_HEIGHT}x{MAX_WIDTH}x{MAX_NUM_FRAMES})"
        )

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check if the service is healthy and models are loaded"""
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Models not loaded"}
        )
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "using_bfloat16": USE_BFLOAT16
    }

@app.post("/generate/text-to-video")
async def generate_text_to_video(params: GenerationParams):
    """Generate video from text prompt"""
    try:
        logger.info(f"Starting text-to-video generation with prompt: {params.prompt}")
        
        # Validate dimensions
        validate_dimensions(params.height, params.width, params.num_frames)
        
        # Set seed
        seed_everething(params.seed)
        generator.manual_seed(params.seed)
        
        # Calculate padded dimensions
        height_padded = ((params.height - 1) // 32 + 1) * 32
        width_padded = ((params.width - 1) // 32 + 1) * 32
        num_frames_padded = ((params.num_frames - 2) // 8 + 1) * 8 + 1
        
        logger.info(f"Using padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}")
        
        # Prepare input
        sample = {
            "prompt": params.prompt,
            "prompt_attention_mask": None,
            "negative_prompt": params.negative_prompt,
            "negative_prompt_attention_mask": None,
            "media_items": None,
        }

        # Generate video
        logger.info("Generating video...")
        images = pipeline(
            num_inference_steps=params.num_inference_steps,
            num_images_per_prompt=1,
            guidance_scale=params.guidance_scale,
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=params.frame_rate,
            **sample,
            is_video=True,
            vae_per_channel_normalize=True,
            conditioning_method=ConditioningMethod.UNCONDITIONAL,
            mixed_precision=params.use_mixed_precision,
        ).images

        # Save video
        output_dir = Path.cwd() / os.getenv("OUTPUT_DIR", "outputs").lstrip("./") / datetime.today().strftime('%Y-%m-%d')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process video frames
        video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        # Get output filename
        output_filename = get_unique_filename(
            "text_to_vid_0",
            ".mp4",
            prompt=params.prompt,
            seed=params.seed,
            resolution=(params.height, params.width, params.num_frames),
            dir=output_dir
        )

        # Write video file
        logger.info(f"Saving video to {output_filename}")
        with imageio.get_writer(output_filename, fps=params.frame_rate) as video:
            for frame in video_np:
                video.append_data(frame)
                
        return FileResponse(
            output_filename,
            media_type="video/mp4",
            filename=output_filename.name
        )

    except Exception as e:
        logger.error(f"Error in text-to-video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/image-to-video")
async def generate_image_to_video(
    file: UploadFile = File(...),
    params: GenerationParams = None
):
    """Generate video from input image and text prompt"""
    try:
        logger.info(f"Starting image-to-video generation with prompt: {params.prompt}")
        
        # Validate dimensions
        validate_dimensions(params.height, params.width, params.num_frames)
        
        # Save uploaded image temporarily
        temp_image_path = f"temp_{file.filename}"
        try:
            with open(temp_image_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info("Processing input image...")
            # Load and process image
            media_items_prepad = load_image_to_tensor_with_resize_and_crop(
                temp_image_path, 
                params.height, 
                params.width
            )
        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        
        # Calculate padded dimensions
        height_padded = ((params.height - 1) // 32 + 1) * 32
        width_padded = ((params.width - 1) // 32 + 1) * 32
        num_frames_padded = ((params.num_frames - 2) // 8 + 1) * 8 + 1
        
        logger.info(f"Using padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}")
        
        # Calculate padding
        padding = calculate_padding(params.height, params.width, height_padded, width_padded)
        
        # Apply padding to media items
        media_items = torch.nn.functional.pad(
            media_items_prepad, 
            padding, 
            mode="constant", 
            value=-1
        )

        # Set seed
        seed_everething(params.seed)
        generator.manual_seed(params.seed)
        
        # Prepare input
        sample = {
            "prompt": params.prompt,
            "prompt_attention_mask": None,
            "negative_prompt": params.negative_prompt,
            "negative_prompt_attention_mask": None,
            "media_items": media_items,
        }

        # Generate video
        logger.info("Generating video...")
        images = pipeline(
            num_inference_steps=params.num_inference_steps,
            num_images_per_prompt=1,
            guidance_scale=params.guidance_scale,
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=params.frame_rate,
            **sample,
            is_video=True,
            vae_per_channel_normalize=True,
            conditioning_method=ConditioningMethod.FIRST_FRAME,
            mixed_precision=params.use_mixed_precision,
        ).images

        # Save video
        output_dir = Path.cwd() / os.getenv("OUTPUT_DIR", "outputs").lstrip("./") / datetime.today().strftime('%Y-%m-%d')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process video frames
        video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        # Get output filename
        output_filename = get_unique_filename(
            "img_to_vid_0",
            ".mp4",
            prompt=params.prompt,
            seed=params.seed,
            resolution=(params.height, params.width, params.num_frames),
            dir=output_dir
        )

        # Write video file
        logger.info(f"Saving video to {output_filename}")
        with imageio.get_writer(output_filename, fps=params.frame_rate) as video:
            for frame in video_np:
                video.append_data(frame)
                
        return FileResponse(
            output_filename,
            media_type="video/mp4",
            filename=output_filename.name
        )

    except Exception as e:
        logger.error(f"Error in image-to-video generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
