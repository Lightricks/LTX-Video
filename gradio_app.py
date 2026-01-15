import gradio as gr
import os
import random
import torch
from pathlib import Path
from datetime import datetime

from ltx_video.inference import infer, InferenceConfig


def set_vram_limit(limit_gb):
    """Set VRAM limit for CUDA devices."""
    if limit_gb and limit_gb > 0 and torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if limit_gb < total_memory:
            fraction = limit_gb / total_memory
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
            print(f"VRAM limited to {limit_gb:.1f}GB ({fraction*100:.1f}% of {total_memory:.1f}GB total)")

# Available pipeline configs
PIPELINE_CONFIGS = {
    "LTX-Video 2B Distilled (Fast, Low VRAM)": "configs/ltxv-2b-0.9.8-distilled.yaml",
    "LTX-Video 2B Distilled FP8 (Faster, Lower VRAM)": "configs/ltxv-2b-0.9.8-distilled-fp8.yaml",
    "LTX-Video 13B Distilled (Better Quality)": "configs/ltxv-13b-0.9.8-distilled.yaml",
    "LTX-Video 13B Distilled FP8 (Better Quality, Lower VRAM)": "configs/ltxv-13b-0.9.8-distilled-fp8.yaml",
    "LTX-Video 13B Dev (Highest Quality)": "configs/ltxv-13b-0.9.8-dev.yaml",
    "LTX-Video 13B Dev FP8 (Highest Quality, Lower VRAM)": "configs/ltxv-13b-0.9.8-dev-fp8.yaml",
}

# Resolution presets
RESOLUTION_PRESETS = {
    "1216x704 (Default 16:9)": (1216, 704),
    "704x1216 (Portrait 9:16)": (704, 1216),
    "960x544 (16:9 Lower)": (960, 544),
    "544x960 (9:16 Lower)": (544, 960),
    "768x768 (Square)": (768, 768),
    "512x512 (Square Small)": (512, 512),
}


def generate_video(
    prompt,
    model_choice,
    resolution_preset,
    num_frames,
    frame_rate,
    seed,
    negative_prompt,
    conditioning_image,
    offload_to_cpu,
    limit_gb,
    progress=gr.Progress()
):
    if not prompt:
        return None, "Please enter a prompt."

    # Set VRAM limit if specified
    set_vram_limit(limit_gb)

    # Get pipeline config
    pipeline_config = PIPELINE_CONFIGS.get(model_choice, "configs/ltxv-2b-0.9.8-distilled.yaml")

    # Get resolution
    width, height = RESOLUTION_PRESETS.get(resolution_preset, (1216, 704))

    # Handle seed
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Ensure num_frames follows the 8n+1 rule
    num_frames = int(num_frames)
    num_frames = ((num_frames - 1) // 8) * 8 + 1
    if num_frames < 9:
        num_frames = 9

    # Create output directory
    output_dir = Path(f"outputs/{datetime.today().strftime('%Y-%m-%d')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup conditioning if image provided
    conditioning_media_paths = None
    conditioning_start_frames = None
    if conditioning_image is not None:
        conditioning_media_paths = [conditioning_image]
        conditioning_start_frames = [0]

    progress(0.1, desc="Loading model...")

    try:
        config = InferenceConfig(
            prompt=prompt,
            output_path=str(output_dir),
            pipeline_config=pipeline_config,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=int(frame_rate),
            negative_prompt=negative_prompt,
            conditioning_media_paths=conditioning_media_paths,
            conditioning_start_frames=conditioning_start_frames,
            offload_to_cpu=offload_to_cpu,
        )

        progress(0.3, desc="Generating video...")
        infer(config=config)

        # Find the most recently created video in output dir
        video_files = list(output_dir.glob("*.mp4"))
        if video_files:
            latest_video = max(video_files, key=os.path.getctime)
            return str(latest_video), f"Video generated successfully! Seed: {seed}"

        # Check for single frame image output
        image_files = list(output_dir.glob("*.png"))
        if image_files:
            latest_image = max(image_files, key=os.path.getctime)
            return str(latest_image), f"Image generated successfully! Seed: {seed}"

        return None, "Generation completed but no output file found."

    except Exception as e:
        return None, f"Error during generation: {str(e)}"


def create_ui():
    with gr.Blocks(title="LTX-Video", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # LTX-Video
            Real-time video generation with the LTX-Video DiT model.
            Supports text-to-video and image-to-video generation.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=3,
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="worst quality, inconsistent motion, blurry, jittery, distorted",
                    lines=2,
                )

                with gr.Row():
                    model_choice = gr.Dropdown(
                        label="Model",
                        choices=list(PIPELINE_CONFIGS.keys()),
                        value="LTX-Video 2B Distilled (Fast, Low VRAM)",
                    )

                with gr.Row():
                    resolution_preset = gr.Dropdown(
                        label="Resolution",
                        choices=list(RESOLUTION_PRESETS.keys()),
                        value="1216x704 (Default 16:9)",
                    )

                with gr.Row():
                    num_frames = gr.Slider(
                        label="Number of Frames",
                        minimum=9,
                        maximum=257,
                        step=8,
                        value=121,
                        info="Will be adjusted to 8n+1 format"
                    )
                    frame_rate = gr.Slider(
                        label="Frame Rate (FPS)",
                        minimum=8,
                        maximum=60,
                        step=1,
                        value=30,
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0,
                    )

                with gr.Accordion("Advanced Options", open=False):
                    conditioning_image = gr.Image(
                        label="Conditioning Image (Optional - for Image-to-Video)",
                        type="filepath",
                    )
                    offload_to_cpu = gr.Checkbox(
                        label="Offload to CPU (for low VRAM GPUs)",
                        value=False,
                    )
                    limit_gb = gr.Slider(
                        label="VRAM Limit (GB)",
                        minimum=0,
                        maximum=48,
                        step=1,
                        value=0,
                        info="Set to 0 for no limit. Limits GPU memory usage."
                    )

                generate_btn = gr.Button("Generate Video", variant="primary")

            with gr.Column(scale=1):
                output_video = gr.Video(label="Generated Video")
                status_text = gr.Textbox(label="Status", interactive=False)

        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                model_choice,
                resolution_preset,
                num_frames,
                frame_rate,
                seed,
                negative_prompt,
                conditioning_image,
                offload_to_cpu,
                limit_gb,
            ],
            outputs=[output_video, status_text],
        )

        gr.Markdown(
            """
            ## Tips
            - **Prompt Engineering**: Focus on detailed, chronological descriptions. Include specific movements, appearances, camera angles, and environmental details.
            - **Model Selection**: Use 2B Distilled for faster generation, 13B models for higher quality.
            - **FP8 Models**: Use FP8 variants if you have limited VRAM (requires compatible GPU).
            - **Image-to-Video**: Upload an image to use as the first frame of your video.
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", share=True)
