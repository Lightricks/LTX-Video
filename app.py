import gradio as gr
from pathlib import Path
import inference

def gradio_interface(
    ckpt_dir: str,
    input_image_path: Path,
    output_path: str,
    seed: int,
    num_inference_steps: int,
    num_images_per_prompt: int,
    guidance_scale: float,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: int,
    bfloat16: bool,
    prompt: str,
    negative_prompt: str,
) -> str:
    """
    Gradio interface wrapper for the video generation pipeline.

    Args:
        (All arguments correspond to the updated run_pipeline function in inference.py)
    
    Returns:
        str: Message indicating success or the error encountered during generation.
    """
    # Prepare arguments as a dictionary
    args = {
        "ckpt_dir": ckpt_dir,
        "input_image_path": input_image_path if input_image_path else None,
        "output_path": output_path if output_path else None,
        "seed": int(seed),
        "num_inference_steps": int(num_inference_steps),
        "num_images_per_prompt": int(num_images_per_prompt),
        "guidance_scale": float(guidance_scale),
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "frame_rate": int(frame_rate),
        "bfloat16": bool(bfloat16),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }

    # Run the pipeline
    try:
        inference.run_pipeline(**args)
        return f"Generation completed. Output saved to: {args['output_path'] or 'outputs/'}"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface setup
with gr.Blocks() as app:
    gr.Markdown("## Video Generation Interface")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt")
        negative_prompt = gr.Textbox(
            label="Negative Prompt", value="worst quality, inconsistent motion, blurry, jittery, distorted"
        )

    with gr.Row():
        ckpt_dir = gr.Textbox(label="Checkpoint Directory", placeholder="Path to model checkpoints", value="PATH")
        input_image_path = gr.File(label="Input Image Path (Optional)", file_types=[".jpg", ".png"])
        output_path = gr.Textbox(
            label="Output Path (Optional)", placeholder="Leave blank to use default output directory"
        )

    with gr.Row():
        seed = gr.Number(label="Seed", value=171198, precision=0)
        num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=1, maximum=100, value=40, step=1)
        num_images_per_prompt = gr.Slider(label="Number of Videos per Prompt", minimum=1, maximum=100, value=1, step=1)
        guidance_scale = gr.Number(label="Guidance Scale", value=3, precision=2)

    with gr.Row():
        height = gr.Slider(label="Height", minimum=64, maximum=1080, value=512, step=32)
        width = gr.Slider(label="Width", minimum=64, maximum=1920, value=768, step=32)
        num_frames = gr.Slider(label="Number of Frames", minimum=1, maximum=300, value=121)
        frame_rate = gr.Slider(label="Frame Rate", minimum=1, maximum=60, value=24)

    bfloat16 = gr.Checkbox(label="Enable bfloat16 Precision", value=False)

    generate_button = gr.Button("Generate Video")
    output_message = gr.Textbox(label="Output Message")

    generate_button.click(
        gradio_interface,
        inputs=[
            ckpt_dir,
            input_image_path,
            output_path,
            seed,
            num_inference_steps,
            num_images_per_prompt,
            guidance_scale,
            height,
            width,
            num_frames,
            frame_rate,
            bfloat16,
            prompt,
            negative_prompt,
        ],
        outputs=output_message,
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)
