import logging
from typing import Union, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

T2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies, When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

I2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies, When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Keep within 150 words.
For best results, build your prompts using this structure:
Describe the image first and then add the user input. Image description should be in first priority! Align to the image caption if it contradicts the user text input.
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Align to the image caption if it contradicts the user text input.
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""


def tensor_to_pil(tensor):
    # Ensure tensor is in range [-1, 1]
    assert tensor.min() >= -1 and tensor.max() <= 1

    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2

    # Rearrange from [C, H, W] to [H, W, C]
    tensor = tensor.permute(1, 2, 0)

    # Convert to numpy array and then to uint8 range [0, 255]
    numpy_image = (tensor.cpu().numpy() * 255).astype("uint8")

    # Convert to PIL Image
    return Image.fromarray(numpy_image)


def generate_cinematic_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompt: Union[str, List[str]],
    conditioning_items: Optional[List] = None,
    max_new_tokens: int = 256,
) -> List[str]:
    prompts = [prompt] if isinstance(prompt, str) else prompt

    if conditioning_items is None:
        prompts = _generate_t2v_prompt(
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            max_new_tokens,
            T2V_CINEMATIC_PROMPT,
        )
    else:
        if len(conditioning_items) > 1 or conditioning_items[0].media_frame_number != 0:
            logger.warning(
                "prompt enhancement does only support unconditional or first frame of conditioning items, returning original prompts"
            )
            return prompts

        first_frame_conditioning_item = conditioning_items[0]
        first_frames = _get_first_frames_from_conditioning_item(
            first_frame_conditioning_item
        )

        assert len(first_frames) == len(
            prompts
        ), "Number of conditioning frames must match number of prompts"

        prompts = _generate_i2v_prompt(
            image_caption_model,
            image_caption_processor,
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            first_frames,
            max_new_tokens,
            I2V_CINEMATIC_PROMPT,
        )

    return prompts


def _get_first_frames_from_conditioning_item(conditioning_item) -> List[Image.Image]:
    frames_tensor = conditioning_item.media_item
    return [
        tensor_to_pil(frames_tensor[i, :, 0, :, :])
        for i in range(frames_tensor.shape[0])
    ]


def _generate_t2v_prompt(
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    system_prompt: str,
) -> List[str]:
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user_prompt: {p}"},
        ]
        for p in prompts
    ]

    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]
    model_inputs = prompt_enhancer_tokenizer(texts, return_tensors="pt").to(
        prompt_enhancer_model.device
    )

    return _generate_and_decode_prompts(
        prompt_enhancer_model, prompt_enhancer_tokenizer, model_inputs, max_new_tokens
    )


def _generate_i2v_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    first_frames: List[Image.Image],
    max_new_tokens: int,
    system_prompt: str,
) -> List[str]:
    image_captions = _generate_image_captions(
        image_caption_model, image_caption_processor, first_frames
    )

    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user_prompt: {p}\nimage_caption: {c}"},
        ]
        for p, c in zip(prompts, image_captions)
    ]

    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]
    model_inputs = prompt_enhancer_tokenizer(texts, return_tensors="pt").to(
        prompt_enhancer_model.device
    )

    return _generate_and_decode_prompts(
        prompt_enhancer_model, prompt_enhancer_tokenizer, model_inputs, max_new_tokens
    )


def _generate_image_captions(
    image_caption_model,
    image_caption_processor,
    images: List[Image.Image],
    system_prompt: str = "<DETAILED_CAPTION>",
) -> List[str]:
    image_caption_prompts = [system_prompt] * len(images)
    inputs = image_caption_processor(
        image_caption_prompts, images, return_tensors="pt"
    ).to(image_caption_model.device)

    with torch.inference_mode():
        generated_ids = image_caption_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

    return image_caption_processor.batch_decode(generated_ids, skip_special_tokens=True)


def _generate_and_decode_prompts(
    prompt_enhancer_model, prompt_enhancer_tokenizer, model_inputs, max_new_tokens: int
) -> List[str]:
    with torch.inference_mode():
        outputs = prompt_enhancer_model.generate(
            **model_inputs, max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
        ]
        decoded_prompts = prompt_enhancer_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    return decoded_prompts
