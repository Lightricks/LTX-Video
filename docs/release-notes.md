# Release Notes

## July, 8th, 2025: New Control Models Released!
- Released three new control models for LTX-Video on HuggingFace:
    * **Depth Control**: [LTX-Video-ICLoRA-depth-13b-0.9.7](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7)
    * **Pose Control**: [LTX-Video-ICLoRA-pose-13b-0.9.7](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7)
    * **Canny Control**: [LTX-Video-ICLoRA-canny-13b-0.9.7](https://huggingface.co/Lightricks/LTX-Video-ICLoRA-canny-13b-0.9.7)


## May, 14th, 2025: New distilled model 13B v0.9.7:
- Release a new 13B distilled model [ltxv-13b-0.9.7-distilled](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled.safetensors)
    * Amazing for iterative work - generates HD videos in 10 seconds, with low-res preview after just 3 seconds (on H100)!
    * Does not require classifier-free guidance and spatio-temporal guidance.
    * Supports sampling with 8 (recommended), or less diffusion steps.
    * Also released a LoRA version of the distilled model, [ltxv-13b-0.9.7-distilled-lora128](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled-lora128.safetensors)
        * Requires only 1GB of VRAM
        * Can be used with the full 13B model for fast inference
- Release a new quantized distilled model [ltxv-13b-0.9.7-distilled-fp8](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled-fp8.safetensors) for *real-time* generation (on H100) with even less VRAM

## May, 5th, 2025: New model 13B v0.9.7:
- Release a new 13B model [ltxv-13b-0.9.7-dev](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-dev.safetensors)
- Release a new quantized model [ltxv-13b-0.9.7-dev-fp8](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-dev-fp8.safetensors) for faster inference with less VRam
- Release a new upscalers
  * [ltxv-temporal-upscaler-0.9.7](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-temporal-upscaler-0.9.7.safetensors)
  * [ltxv-spatial-upscaler-0.9.7](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-spatial-upscaler-0.9.7.safetensors)
- Breakthrough prompt adherence and physical understanding.
- New Pipeline for multi-scale video rendering for fast and high quality results


## April, 15th, 2025: New checkpoints v0.9.6:
- Release a new checkpoint [ltxv-2b-0.9.6-dev-04-25](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.6-dev-04-25.safetensors) with improved quality
- Release a new distilled model [ltxv-2b-0.9.6-distilled-04-25](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.6-distilled-04-25.safetensors)
    * 15x faster inference than non-distilled model.
    * Does not require classifier-free guidance and spatio-temporal guidance.
    * Supports sampling with 8 (recommended), or less diffusion steps.
- Improved prompt adherence, motion quality and fine details.
- New default resolution and FPS: 1216 × 704 pixels at 30 FPS
    * Still real time on H100 with the distilled model.
    * Other resolutions and FPS are still supported.
- Support stochastic inference (can improve visual quality when using the distilled model)

## March, 5th, 2025: New checkpoint v0.9.5
- New license for commercial use ([OpenRail-M](https://huggingface.co/Lightricks/LTX-Video/ltx-video-2b-v0.9.5.license.txt))
- Release a new checkpoint v0.9.5 with improved quality
- Support keyframes and video extension
- Support higher resolutions
- Improved prompt understanding
- Improved VAE
- New online web app in [LTX-Studio](https://app.ltx.studio/ltx-video)
- Automatic prompt enhancement

## February, 20th, 2025: More inference options
- Improve STG (Spatiotemporal Guidance) for LTX-Video
- Support MPS on macOS with PyTorch 2.3.0
- Add support for 8-bit model, LTX-VideoQ8
- Add TeaCache for LTX-Video
- Add [ComfyUI-LTXTricks](#comfyui-integration)
- Add Diffusion-Pipe

## December 31st, 2024: Research paper
- Release the [research paper](https://arxiv.org/abs/2501.00103)

## December 20th, 2024: New checkpoint v0.9.1
- Release a new checkpoint v0.9.1 with improved quality
- Support for STG / PAG
- Support loading checkpoints of LTX-Video in Diffusers format (conversion is done on-the-fly)
- Support offloading unused parts to CPU
- Support the new timestep-conditioned VAE decoder
- Reference contributions from the community in the readme file
- Relax transformers dependency

## November 21th, 2024: Initial release v0.9.0
- Initial release of LTX-Video
- Support text-to-video and image-to-video generation