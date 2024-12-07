import gradio as gr
from pathlib import Path
import inference
import json
import random

from typing import List

SETTINGS_FILE = "settings.json"

def load_settings():
    """Load settings from `settings.json` if it exists."""
    if Path(SETTINGS_FILE).is_file():
        with open(SETTINGS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}

def save_settings(settings):
    """Save settings to `settings.json`."""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=4)

def build_prompt(age="middle aged", race="generic", gender="female", hair_length="short", hair_texture="curly", 
                 hair_color="dark", object_a="lamp", object_b="plant", clothing_tone="dark", clothing_type="suit", 
                 shirt_color="white", shirt_type="dress shirt", accessory="tie", broadcast_type="news"):
    # Derive pronouns from gender. Ideally, we'd have something more nuanced to support 
    # more ambiguously gendered / androgynous avatars but that's poorly represented in
    # the datasets and wildly hit-or-miss for generation
    if (gender=="man"):
        pronoun="he"
        poss_pronoun="his"
    else:
        pronoun="she"
        poss_pronoun="her"
    
    hair_prompt=f"A {age} {race} {gender} with {hair_length} {hair_color} hair"
    if (hair_length=="shaved") & (hair_texture=="bald"):
        hair_prompt=f"A {age} {race} {gender} with a {hair_length} head"

    gen_prompt=f"{hair_prompt} looks directly into the camera. The background is out of focus and contains a {object_a} and a {object_b}. {pronoun}'s wearing a {clothing_tone} {clothing_type} with a {shirt_color} {shirt_type} and {accessory}. {pronoun} blinks and nods {poss_pronoun} head and looks intently into the camera. The camera remains stationary framing {poss_pronoun} head and shoulders. High quality professional lighting. The scene appears to be from a {broadcast_type} broadcast."
    return gen_prompt

def custom_sort_key(item: str) -> str:
    """
    Sorting key function that skips the prefixes 'a ', 'an ', and 'the ' if present.
    """
    # Define a list of prefixes to ignore
    prefixes = ["a ", "an ", "the "]
    
    # Check if the string starts with any of the prefixes (case-insensitive)
    for prefix in prefixes:
        if item.lower().startswith(prefix):
            return item[len(prefix):].lower()  # Remove the prefix for sorting key
    
    # If no prefix matches, return the whole string
    return item.lower()

def sort_custom(items: List[str]) -> List[str]:
    """
    Sorts a list of strings alphabetically, but ignores the prefixes 'a ', 'an ', and 'the '.
    """
    return sorted(items, key=custom_sort_key)

def gradio_interface(
    seed,
    num_inference_steps,
    num_images_per_prompt,
    height,
    width,
    num_runs,
    num_frames,
    frame_rate,
    extend_clip,
    restart_first_frame,
    upscale_clip,
    age,
    race,
    gender,
    hair_length,
    hair_texture,
    hair_color,
    object_a,
    object_b,
    clothing_tone,
    clothing_type,
    shirt_color,
    shirt_type,
    accessory,
    broadcast_type
) -> str:
    """
    Gradio interface wrapper for the video generation pipeline.
    """
    # Prepare arguments as a dictionary
    args = {
        "ckpt_dir": "PATH",
        "input_image_path": None,
        "output_path": None,
        "seed": int(seed),
        "num_inference_steps": int(num_inference_steps),
        "num_runs": int(num_runs),
        "num_images_per_prompt": int(num_images_per_prompt),
        "guidance_scale": 3.0,
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "frame_rate": int(frame_rate),
        "bfloat16": False,
        "prompt": build_prompt(age=age, race=race, gender=gender, hair_length=hair_length, 
                               hair_texture=hair_texture, hair_color=hair_color, object_a=object_a,
                               object_b=object_b, clothing_tone=clothing_tone, clothing_type=clothing_type,
                               shirt_color=shirt_color, shirt_type=shirt_type, accessory=accessory, broadcast_type=broadcast_type),
        "negative_prompt": "text, logo, graphics, worst quality, deformed, distorted, inconsistent motion, blurry, jittery, distorted",
        "extend_clip": extend_clip,
        "restart_first_frame": restart_first_frame,
        "upscale_clip": upscale_clip,
        "override_filename": f"{age}_{race}_{gender}_{hair_length}_{hair_texture}_{hair_color}"
    }

    # Save current settings
    save_settings(args)

    # Run the pipeline
    try:
        inference.run_pipeline(**args)
        return f"Avatar Generation complete."
    except Exception as e:
        return f"Error: {str(e)}"

# Load default settings
default_settings = load_settings()

options={
    'age'           :   ['young', 'middle-aged', 'older', 'elderly'],
    'race'          :   ['white', 'black', 'latino', 'asian'],
    'gender'        :   ['man','woman'],
    'hair_length'   :   ['shaved','short','long'],
    'hair_texture'  :   ['straight', 'curly', 'wavy', 'bald'],
    'hair_color'    :   ["black", "brown", "blonde", "red", "auburn", "gray", "white", "blue", "green", "pink", "purple", "silver", "platinum", "strawberry blonde", "chestnut"],
    'objects'      :    ["a bookshelf", "picture frames", "plants", "a lamp", "a clock", "curtains", "awards", "a vase", "artwork", "a window", "a television screen", "a computer monitor", "a couch", "a chair", "a desk", "a microphone", "a whiteboard", "books", "trophies", "decorative figurines", "a news desk", "television monitors", "digital screens", "a world map", "a weather map", "cityscape backdrop", "camera equipment", "lighting rigs", "a flag", "an American flag", "a Pride flag", "charts", "newsroom activity", "computer screens", "studio decor", "a globe"],
    'clothing_tone' :   ['dark', 'light', 'tan', 'desert camo', 'woodland camo', 'digital camo'],
    'clothing_type' :   ['suit', 'casual outfit', 'workout outfit', 'police uniform', 'firefighter uniform', 'uniform'],
    'shirt_color'   :   ["white", "black", "blue", "red", "green", "yellow", "gray", "brown", "pink", "purple", "orange", "beige", "maroon", "navy", "teal", "turquoise", "lavender", "olive", "gold", "silver"],
    'shirt_type'    :   ["shirt", "t-shirt", "dress shirt", "polo shirt"],
    'accessory'     :   ['tie', 'bow', 'bow tie', 'glasses', 'sunglasses', 'hat', 'ballcap', 'tophat', 'trucker hat', 'tattoos', 'lapel flag pin', 'pony tail', 'headphones', 'headset', 'microphone', 'clean face'],
    'broadcast_type':   ['news', 'sports', 'comedy', 'weather', 'documentary', 'awards']
}

for key in options:
    options[key]=sort_custom(options[key])

# Gradio interface setup
with gr.Blocks() as app:
    gr.Markdown("# AuRA Avatar Generation\n*from Big Blue Ceiling*")

    with gr.Row():
        with gr.Column(scale=1):
            age=gr.Dropdown(choices=options['age'], label="Age", value=random.choice(options['age']))
            race=gr.Dropdown(choices=options['race'], label="Race", value=random.choice(options['race']))
            gender=gr.Dropdown(choices=options['gender'], label="Gender", value=random.choice(options['gender']))
            hair_length=gr.Dropdown(choices=options['hair_length'], label="Hair length", value=random.choice(options['hair_length']))
            hair_texture=gr.Dropdown(choices=options['hair_texture'], label="Hair texture", value=random.choice(options['hair_texture']))
            hair_color=gr.Dropdown(choices=options['hair_color'], label="Hair color", value=random.choice(options['hair_color']))
            object_a=gr.Dropdown(choices=options['objects'], label="BG Object A", value=random.choice(options['objects']))
        with gr.Column(scale=1):
            object_b=gr.Dropdown(choices=options['objects'], label="BG Object B", value=random.choice(options['objects']))
            clothing_tone=gr.Dropdown(choices=options['clothing_tone'], label="Clothing tone", value=random.choice(options['clothing_tone']))
            clothing_type=gr.Dropdown(choices=options['clothing_type'], label="Clothing type", value=random.choice(options['clothing_type']))
            shirt_color=gr.Dropdown(choices=options['shirt_color'], label="Shirt color", value=random.choice(options['shirt_color']))
            shirt_type=gr.Dropdown(choices=options['shirt_type'], label="Shirt type", value=random.choice(options['shirt_type']))
            accessory=gr.Dropdown(choices=options['accessory'], label="Accessory", value=random.choice(options['accessory']))
            broadcast_type=gr.Dropdown(choices=options['broadcast_type'], label="Broadcast type")
        with gr.Column(scale=3):
            with gr.Row():
                seed = gr.Number(
                    label="Seed", 
                    value=default_settings.get("seed", -1), 
                    precision=0
                )
                num_inference_steps = gr.Slider(
                    label="Number of Inference Steps", 
                    minimum=1, 
                    maximum=100, 
                    value=default_settings.get("num_inference_steps", 40), 
                    step=1,
                    visible=False
                )
                num_runs = gr.Slider(
                    label="Number of runs to generate", 
                    minimum=1, 
                    maximum=100, 
                    value=default_settings.get("num_runs", 1), 
                    step=1
                )
                num_images_per_prompt = gr.Slider(
                    label="Number of avatar videos per run", 
                    minimum=1, 
                    maximum=100, 
                    value=default_settings.get("num_images_per_prompt", 1), 
                    step=1
                )

            with gr.Row():
                height = gr.Slider(
                    label="Height", 
                    minimum=64, 
                    maximum=1080, 
                    value=default_settings.get("height", 416), 
                    step=32, 
                    visible=False
                )
                width = gr.Slider(
                    label="Width", 
                    minimum=64, 
                    maximum=1920, 
                    value=default_settings.get("width", 768), 
                    step=32, 
                    visible=False
                )
                num_frames = gr.Slider(
                    label="Number of Frames", 
                    minimum=1, 
                    maximum=300, 
                    value=default_settings.get("num_frames", 121),
                    visible=False
                )
                frame_rate = gr.Slider(
                    label="Frame Rate", 
                    minimum=1, 
                    maximum=60, 
                    value=default_settings.get("frame_rate", 24)
                )

            with gr.Row():
                extendClip = gr.Checkbox(
                    label="Create extended clip series", 
                    value=default_settings.get("extend_clip", True)
                )
                restartFirstFrame = gr.Checkbox(
                    label="Restart extended clip with first frame", 
                    value=default_settings.get("restart_first_frame", False)
                )
                upscale_clip = gr.Checkbox(
                    label="Upscale", 
                    value=default_settings.get("upscale_clip", False)
                )

            generate_button = gr.Button("Generate Avatars")
    output_message = gr.Textbox(label="Output Message")

    generate_button.click(
        gradio_interface,
        inputs=[
            seed,
            num_inference_steps,
            num_images_per_prompt,
            height,
            width,
            num_runs,
            num_frames,
            frame_rate,
            extendClip,
            restartFirstFrame,
            upscale_clip,
            age,
            race,
            gender,
            hair_length,
            hair_texture,
            hair_color,
            object_a,
            object_b,
            clothing_tone,
            clothing_type,
            shirt_color,
            shirt_type,
            accessory,
            broadcast_type
        ],
        outputs=output_message,
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)
