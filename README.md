<div align="center">

# LTX-Video

[![Website](https://img.shields.io/badge/Website-LTXV-181717?logo=google-chrome)](https://www.lightricks.com/ltxv)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-Video)
[![Demo](https://img.shields.io/badge/Demo-Try%20Now-brightgreen?logo=vercel)](https://app.ltx.studio/motion-workspace?videoModel=ltxv-13b)
[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv)](https://arxiv.org/abs/2501.00103)
[![Trainer](https://img.shields.io/badge/LTXV-Trainer-9146FF?logo=github)](https://github.com/Lightricks/LTX-Video-Trainer)
[![Discord](https://img.shields.io/badge/Join-Discord-5865F2?logo=discord)](https://discord.gg/Mn8BRgUKKy)

This is the official repository for LTX-Video.

</div>

## Table of Contents

- [Introduction](#introduction)
- [What's new](./docs/release-notes.md)
- [Models](#models)
- [Quick Start Guide](#quick-start-guide)
  - [Online demo](#online-inference)
  - [Run locally](#run-locally)
    - [Installation](#installation)
    - [Inference](#inference)
  - [ComfyUI Integration](#comfyui-integration)
  - [Diffusers Integration](#diffusers-integration)
- [Model User Guide](#model-user-guide)
- [Community Contribution](#community-contribution)
- [Training](#training)
- [Control Models](#control-models)
- [Join Us!](#join-us-)
- [Acknowledgement](#acknowledgement)

# Introduction

LTX-Video is the first DiT-based video generation model that can generate high-quality videos in *real-time*.
It can generate 30 FPS videos at 1216×704 resolution, faster than it takes to watch them.
The model is trained on a large-scale dataset of diverse videos and can generate high-resolution videos
with realistic and diverse content.

The model supports image-to-video, keyframe-based animation, video extension (both forward and backward), video-to-video transformations, and any combination of these features.

### Image-to-video examples
| | | |
|:---:|:---:|:---:|
| ![example1](./docs/_static/ltx-video_i2v_example_00001.gif) | ![example2](./docs/_static/ltx-video_i2v_example_00002.gif) | ![example3](./docs/_static/ltx-video_i2v_example_00003.gif) |
| ![example4](./docs/_static/ltx-video_i2v_example_00004.gif) | ![example5](./docs/_static/ltx-video_i2v_example_00005.gif) |  ![example6](./docs/_static/ltx-video_i2v_example_00006.gif) |
| ![example7](./docs/_static/ltx-video_i2v_example_00007.gif) |  ![example8](./docs/_static/ltx-video_i2v_example_00008.gif) | ![example9](./docs/_static/ltx-video_i2v_example_00009.gif) |

### Controlled video examples
| | | |
|:---:|:---:|:---:|
| ![control0](./docs/_static/ltx-video_ic_2v_example_00000.gif) | ![control1](./docs/_static/ltx-video_ic_2v_example_00001.gif) | ![control2](./docs/_static/ltx-video_ic_2v_example_00002.gif) |

| | |
|:---:|:---:|
| ![control3](./docs/_static/ltx-video_ic_2v_example_00003.gif) | ![control4](./docs/_static/ltx-video_ic_2v_example_00004.gif) |


# Models & Workflows

| Name                    | Notes                                                                                      | inference.py config                                                                                                                                      | ComfyUI workflow (Recommended) |
|-------------------------|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| ltxv-13b-0.9.7-dev                   | Highest quality, requires more VRAM                                                        | [ltxv-13b-0.9.7-dev.yaml](https://github.com/Lightricks/LTX-Video/blob/main/configs/ltxv-13b-0.9.7-dev.yaml)                                             | [ltxv-13b-i2v-base.json](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/ltxv-13b-i2v-base.json)             |
| [ltxv-13b-0.9.7-mix](https://app.ltx.studio/motion-workspace?videoModel=ltxv-13b)            | Mix ltxv-13b-dev and ltxv-13b-distilled in the same multi-scale rendering workflow for balanced speed-quality | N/A                                             | [ltxv-13b-i2v-mixed-multiscale.json](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/ltxv-13b-i2v-mixed-multiscale.json)             |
 [ltxv-13b-0.9.7-distilled](https://app.ltx.studio/motion-workspace?videoModel=ltxv)        | Faster, less VRAM usage, slight quality reduction compared to 13b. Ideal for rapid iterations | [ltxv-13b-0.9.7-distilled.yaml](https://github.com/Lightricks/LTX-Video/blob/main/configs/ltxv-13b-0.9.7-dev.yaml)                                    | [ltxv-13b-dist-i2v-base.json](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/13b-distilled/ltxv-13b-dist-i2v-base.json) |
| [ltxv-13b-0.9.7-distilled-lora128](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled-lora128.safetensors)         | LoRA to make ltxv-13b-dev behave like the distilled model | N/A                                    | N/A |
| ltxv-13b-0.9.7-dev-fp8               | Quantized version of ltxv-13b | [ltxv-13b-0.9.7-dev-fp8.yaml](https://github.com/Lightricks/LTX-Video/blob/main/configs/ltxv-13b-0.9.7-dev-fp8.yaml) | [ltxv-13b-i2v-base-fp8.json](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/ltxv-13b-i2v-base-fp8.json) |
| ltxv-13b-0.9.7-distilled-fp8     | Quantized version of ltxv-13b-distilled | [ltxv-13b-0.9.7-distilled-fp8.yaml](https://github.com/Lightricks/LTX-Video/blob/main/configs/ltxv-13b-0.9.7-distilled-fp8.yaml) | [ltxv-13b-dist-i2v-base-fp8.json](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/13b-distilled/ltxv-13b-dist-i2v-base-fp8.json) |
| ltxv-2b-0.9.6                     | Good quality, lower VRAM requirement than ltxv-13b                                         | [ltxv-2b-0.9.6-dev.yaml](https://github.com/Lightricks/LTX-Video/blob/main/configs/ltxv-2b-0.9.6-dev.yaml)                                                 | [ltxvideo-i2v.json](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/low_level/ltxvideo-i2v.json)             |
| ltxv-2b-0.9.6-distilled         | 15× faster, real-time capable, fewer steps needed, no STG/CFG required                     | [ltxv-2b-0.9.6-distilled.yaml](https://github.com/Lightricks/LTX-Video/blob/main/configs/ltxv-2b-0.9.6-distilled.yaml)                                     | [ltxvideo-i2v-distilled.json](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/low_level/ltxvideo-i2v-distilled.json)             |


# Quick Start Guide

## Online inference
The model is accessible right away via the following links:
- [LTX-Studio image-to-video (13B-mix)](https://app.ltx.studio/motion-workspace?videoModel=ltxv-13b)
- [LTX-Studio image-to-video (13B distilled)](https://app.ltx.studio/motion-workspace?videoModel=ltxv)
- [Fal.ai image-to-video (13B full)](https://fal.ai/models/fal-ai/ltx-video-13b-dev/image-to-video)
- [Fal.ai image-to-video (13B distilled)](https://fal.ai/models/fal-ai/ltx-video-13b-distilled/image-to-video)
- [Replicate image-to-video](https://replicate.com/lightricks/ltx-video)

## Run locally

### Installation
The codebase was tested with Python 3.10.5, CUDA version 12.2, and supports PyTorch >= 2.1.2.
On macOS, MPS was tested with PyTorch 2.3.0, and should support PyTorch == 2.3 or >= 2.6.

```bash
git clone https://github.com/Lightricks/LTX-Video.git
cd LTX-Video

# create env
python -m venv env
source env/bin/activate
python -m pip install -e \[inference\]
```

#### FP8 Kernels (optional)

[FP8 kernels](https://github.com/Lightricks/LTXVideo-Q8-Kernels) developed for LTX-Video provide performance boost on supported graphics cards (Ada architecture and later). To install FP8 kernels, follow the instructions in that repository.

### Inference

📝 **Note:** For best results, we recommend using our [ComfyUI](#comfyui-integration) workflow. We're working on updating the inference.py script to match the high quality and output fidelity of ComfyUI.

To use our model, please follow the inference code in [inference.py](./inference.py):

#### For image-to-video generation:

```bash
python inference.py --prompt "PROMPT" --conditioning_media_paths IMAGE_PATH --conditioning_start_frames 0 --height HEIGHT --width WIDTH --num_frames NUM_FRAMES --seed SEED --pipeline_config configs/ltxv-13b-0.9.7-distilled.yaml
```

#### Extending a video:

📝 **Note:** Input video segments must contain a multiple of 8 frames plus 1 (e.g., 9, 17, 25, etc.), and the target frame number should be a multiple of 8.


```bash
python inference.py --prompt "PROMPT" --conditioning_media_paths VIDEO_PATH --conditioning_start_frames START_FRAME --height HEIGHT --width WIDTH --num_frames NUM_FRAMES --seed SEED --pipeline_config configs/ltxv-13b-0.9.7-distilled.yaml
```

#### For video generation with multiple conditions:

You can now generate a video conditioned on a set of images and/or short video segments.
Simply provide a list of paths to the images or video segments you want to condition on, along with their target frame numbers in the generated video. You can also specify the conditioning strength for each item (default: 1.0).

```bash
python inference.py --prompt "PROMPT" --conditioning_media_paths IMAGE_OR_VIDEO_PATH_1 IMAGE_OR_VIDEO_PATH_2 --conditioning_start_frames TARGET_FRAME_1 TARGET_FRAME_2 --height HEIGHT --width WIDTH --num_frames NUM_FRAMES --seed SEED --pipeline_config configs/ltxv-13b-0.9.7-distilled.yaml
```

### Using as a library

```python
from ltx_video.inference import infer, InferenceConfig

infer(
    InferenceConfig(
        pipeline_config="configs/ltxv-13b-0.9.7-distilled.yaml",
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        output_path="output.mp4",
    )
)
```

## ComfyUI Integration
To use our model with ComfyUI, please follow the instructions at [https://github.com/Lightricks/ComfyUI-LTXVideo/](https://github.com/Lightricks/ComfyUI-LTXVideo/).

## Diffusers Integration
To use our model with the Diffusers Python library, check out the [official documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video).

Diffusers also support an 8-bit version of LTX-Video, [see details below](#ltx-videoq8)

# Model User Guide

## 📝 Prompt Engineering

When writing prompts, focus on detailed, chronological descriptions of actions and scenes. Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. Start directly with the action, and keep descriptions literal and precise. Think like a cinematographer describing a shot list. Keep within 200 words. For best results, build your prompts using this structure:

* Start with main action in a single sentence
* Add specific details about movements and gestures
* Describe character/object appearances precisely
* Include background and environment details
* Specify camera angles and movements
* Describe lighting and colors
* Note any changes or sudden events
* See [examples](#introduction) for more inspiration.

### Automatic Prompt Enhancement

When using `LTXVideoPipeline` directly, you can enable prompt enhancement by setting `enhance_prompt=True`.

## 🎮 Parameter Guide

* Resolution Preset: Higher resolutions for detailed scenes, lower for faster generation and simpler scenes. The model works on resolutions that are divisible by 32 and number of frames that are divisible by 8 + 1 (e.g. 257). In case the resolution or number of frames are not divisible by 32 or 8 + 1, the input will be padded with -1 and then cropped to the desired resolution and number of frames. The model works best on resolutions under 720 x 1280 and number of frames below 257
* Seed: Save seed values to recreate specific styles or compositions you like
* Guidance Scale: 3-3.5 are the recommended values
* Inference Steps: More steps (40+) for quality, fewer steps (20-30) for speed

📝 For advanced parameters usage, please see `python inference.py --help`

## Community Contribution

### ComfyUI-LTXTricks 🛠️

A community project providing additional nodes for enhanced control over the LTX Video model. It includes implementations of advanced techniques like RF-Inversion, RF-Edit, FlowEdit, and more. These nodes enable workflows such as Image and Video to Video (I+V2V), enhanced sampling via Spatiotemporal Skip Guidance (STG), and interpolation with precise frame settings.

- **Repository:** [ComfyUI-LTXTricks](https://github.com/logtd/ComfyUI-LTXTricks)
- **Features:**
  - 🔄 **RF-Inversion:** Implements [RF-Inversion](https://rf-inversion.github.io/) with an [example workflow here](https://github.com/logtd/ComfyUI-LTXTricks/blob/main/example_workflows/example_ltx_inversion.json).
  - ✂️ **RF-Edit:** Implements [RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit) with an [example workflow here](https://github.com/logtd/ComfyUI-LTXTricks/blob/main/example_workflows/example_ltx_rf_edit.json).
  - 🌊 **FlowEdit:** Implements [FlowEdit](https://github.com/fallenshock/FlowEdit) with an [example workflow here](https://github.com/logtd/ComfyUI-LTXTricks/blob/main/example_workflows/example_ltx_flow_edit.json).
  - 🎥 **I+V2V:** Enables Video to Video with a reference image. [Example workflow](https://github.com/logtd/ComfyUI-LTXTricks/blob/main/example_workflows/example_ltx_iv2v.json).
  - ✨ **Enhance:** Partial implementation of [STGuidance](https://junhahyung.github.io/STGuidance/). [Example workflow](https://github.com/logtd/ComfyUI-LTXTricks/blob/main/example_workflows/example_ltxv_stg.json).
  - 🖼️ **Interpolation and Frame Setting:** Nodes for precise control of latents per frame. [Example workflow](https://github.com/logtd/ComfyUI-LTXTricks/blob/main/example_workflows/example_ltx_interpolation.json).


### LTX-VideoQ8 🎱 <a id="ltx-videoq8"></a>

**LTX-VideoQ8** is an 8-bit optimized version of [LTX-Video](https://github.com/Lightricks/LTX-Video), designed for faster performance on NVIDIA ADA GPUs.

- **Repository:** [LTX-VideoQ8](https://github.com/KONAKONA666/LTX-Video)
- **Features:**
  - 🚀 Up to 3X speed-up with no accuracy loss
  - 🎥 Generate 720x480x121 videos in under a minute on RTX 4060 (8GB VRAM)
  - 🛠️ Fine-tune 2B transformer models with precalculated latents
- **Community Discussion:** [Reddit Thread](https://www.reddit.com/r/StableDiffusion/comments/1h79ks2/fast_ltx_video_on_rtx_4060_and_other_ada_gpus/)
- **Diffusers integration:** A diffusers integration for the 8-bit model is already out! [Details here](https://github.com/sayakpaul/q8-ltx-video)


### TeaCache for LTX-Video 🍵 <a id="TeaCache"></a>

**TeaCache** is a training-free caching approach that leverages timestep differences across model outputs to accelerate LTX-Video inference by up to 2x without significant visual quality degradation.

- **Repository:** [TeaCache4LTX-Video](https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4LTX-Video)
- **Features:**
  - 🚀 Speeds up LTX-Video inference.
  - 📊 Adjustable trade-offs between speed (up to 2x) and visual quality using configurable parameters.
  - 🛠️ No retraining required: Works directly with existing models.

### Your Contribution

...is welcome! If you have a project or tool that integrates with LTX-Video,
please let us know by opening an issue or pull request.

# ⚡️ Training

We provide an open-source repository for fine-tuning the LTX-Video model: [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer).
This repository supports both the 2B and 13B model variants, enabling full fine-tuning as well as LoRA (Low-Rank Adaptation) fine-tuning for more efficient training. This includes:

- **Control LoRAs**: Train custom control models like depth, pose, and canny control
- **Effect LoRAs**: Create specialized effects and transformations for video generation

Explore the repository to customize the model for your specific use cases!
More information and training instructions can be found in the [README](https://github.com/Lightricks/LTX-Video-Trainer/blob/main/README.md).

# 🎬 Control Models

[ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) repository now contains workflows and models for 3 specialized models that enable precise control over LTX-Video generation:

Pose Control, Depth Control and Canny Control

**Example ComfyUI Workflow (for all control types):** [ic-lora.json](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/ic_lora/ic-lora.json)

# 🚀 Join Us

Want to work on cutting-edge AI research and make a real impact on millions of users worldwide?

At **Lightricks**, an AI-first company, we're revolutionizing how visual content is created.

If you are passionate about AI, computer vision, and video generation, we would love to hear from you!

Please visit our [careers page](https://careers.lightricks.com/careers?query=&office=all&department=R%26D) for more information.

# Acknowledgement

We are grateful for the following awesome projects when implementing LTX-Video:
* [DiT](https://github.com/facebookresearch/DiT) and [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha): vision transformers for image generation.


## Citation

📄 Our tech report is out! If you find our work helpful, please ⭐️ star the repository and cite our paper.

```
@article{HaCohen2024LTXVideo,
  title={LTX-Video: Realtime Video Latent Diffusion},
  author={HaCohen, Yoav and Chiprut, Nisan and Brazowski, Benny and Shalem, Daniel and Moshe, Dudu and Richardson, Eitan and Levin, Eran and Shiran, Guy and Zabari, Nir and Gordon, Ori and Panet, Poriya and Weissbuch, Sapir and Kulikov, Victor and Bitterman, Yaki and Melumian, Zeev and Bibi, Ofir},
  journal={arXiv preprint arXiv:2501.00103},
  year={2024}
}
```
