import pytest
import torch
import yaml
from inference import infer, create_ltx_video_pipeline
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, str):
        return f"{argname}-{val}"
    return f"{argname}-{repr(val)}"


@pytest.mark.parametrize(
    "conditioning_test_mode",
    ["unconditional", "first-frame", "first-sequence", "sequence-and-frame"],
    ids=lambda x: f"conditioning_test_mode={x}",
)
def test_infer_runs_on_real_path(tmp_path, test_paths, conditioning_test_mode):
    conditioning_params = {}
    if conditioning_test_mode == "unconditional":
        pass
    elif conditioning_test_mode == "first-frame":
        conditioning_params["conditioning_media_paths"] = [
            test_paths["input_image_path"]
        ]
        conditioning_params["conditioning_start_frames"] = [0]
    elif conditioning_test_mode == "first-sequence":
        conditioning_params["conditioning_media_paths"] = [
            test_paths["input_video_path"]
        ]
        conditioning_params["conditioning_start_frames"] = [0]
    elif conditioning_test_mode == "sequence-and-frame":
        conditioning_params["conditioning_media_paths"] = [
            test_paths["input_video_path"],
            test_paths["input_image_path"],
        ]
        conditioning_params["conditioning_start_frames"] = [16, 67]
    else:
        raise ValueError(f"Unknown conditioning mode: {conditioning_test_mode}")
    test_paths = {
        k: v
        for k, v in test_paths.items()
        if k not in ["input_image_path", "input_video_path"]
    }

    params = {
        "seed": 42,
        "num_inference_steps": 1,
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "frame_rate": 25,
        "prompt": "A young woman with wavy, shoulder-length light brown hair stands outdoors on a foggy day. She wears a cozy pink turtleneck sweater, with a serene expression and piercing blue eyes. A wooden fence and a misty, grassy field fade into the background, evoking a calm and introspective mood.",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "offload_to_cpu": False,
        "output_path": tmp_path,
        "image_cond_noise_scale": 0.15,
    }

    config = {
        "pipeline_type": "base",
        "num_images_per_prompt": 1,
        "guidance_scale": 2.5,
        "stg_scale": 1,
        "stg_rescale": 0.7,
        "stg_mode": "attention_values",
        "stg_skip_layers": "1,2,3",
        "precision": "bfloat16",
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
        "checkpoint_path": test_paths["ckpt_path"],
        "text_encoder_model_name_or_path": test_paths[
            "text_encoder_model_name_or_path"
        ],
        "prompt_enhancer_image_caption_model_name_or_path": test_paths[
            "prompt_enhancer_image_caption_model_name_or_path"
        ],
        "prompt_enhancer_llm_model_name_or_path": test_paths[
            "prompt_enhancer_llm_model_name_or_path"
        ],
        "prompt_enhancement_words_threshold": 120,
        "stochastic_sampling": False,
        "sampler": "from_checkpoint",
    }

    temp_config_path = tmp_path / "config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    infer(**{**conditioning_params, **params, "pipeline_config": temp_config_path})


def test_vid2vid(tmp_path, test_paths):
    params = {
        "seed": 42,
        "image_cond_noise_scale": 0.15,
        "height": 512,
        "width": 768,
        "num_frames": 25,
        "frame_rate": 25,
        "prompt": "A young woman with wavy, shoulder-length light brown hair stands outdoors on a foggy day. She wears a cozy pink turtleneck sweater, with a serene expression and piercing blue eyes. A wooden fence and a misty, grassy field fade into the background, evoking a calm and introspective mood.",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "offload_to_cpu": False,
        "input_media_path": test_paths["input_video_path"],
    }

    config = {
        "num_inference_steps": 3,
        "skip_initial_inference_steps": 1,
        "guidance_scale": 2.5,
        "stg_scale": 1,
        "stg_rescale": 0.7,
        "stg_mode": "attention_values",
        "stg_skip_layers": "1,2,3",
        "precision": "bfloat16",
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
        "sampler": "from_checkpoint",
        "checkpoint_path": test_paths["ckpt_path"],
        "text_encoder_model_name_or_path": test_paths[
            "text_encoder_model_name_or_path"
        ],
        "prompt_enhancer_image_caption_model_name_or_path": test_paths[
            "prompt_enhancer_image_caption_model_name_or_path"
        ],
        "prompt_enhancer_llm_model_name_or_path": test_paths[
            "prompt_enhancer_llm_model_name_or_path"
        ],
        "prompt_enhancement_words_threshold": 120,
    }
    test_paths = {
        k: v
        for k, v in test_paths.items()
        if k not in ["input_image_path", "input_video_path"]
    }
    temp_config_path = tmp_path / "config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    infer(**{**test_paths, **params, "pipeline_config": temp_config_path})


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def test_pipeline_on_batch(tmp_path, test_paths):
    device = get_device()
    pipeline = create_ltx_video_pipeline(
        ckpt_path=test_paths["ckpt_path"],
        device=device,
        precision="bfloat16",
        text_encoder_model_name_or_path=test_paths["text_encoder_model_name_or_path"],
        enhance_prompt=False,
        prompt_enhancer_image_caption_model_name_or_path=test_paths[
            "prompt_enhancer_image_caption_model_name_or_path"
        ],
        prompt_enhancer_llm_model_name_or_path=test_paths[
            "prompt_enhancer_llm_model_name_or_path"
        ],
    )

    params = {
        "seed": 42,
        "image_cond_noise_scale": 0.15,
        "height": 512,
        "width": 768,
        "num_frames": 1,
        "frame_rate": 25,
        "offload_to_cpu": False,
        "output_type": "pt",
        "is_video": False,
        "vae_per_channel_normalize": True,
        "mixed_precision": False,
    }

    config = {
        "num_inference_steps": 2,
        "guidance_scale": 2.5,
        "stg_scale": 1,
        "rescaling_scale": 0.7,
        "skip_block_list": [1, 2],
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
    }

    temp_config_path = tmp_path / "config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    first_prompt = "A vintage yellow car drives along a wet mountain road, its rear wheels kicking up a light spray as it moves. The camera follows close behind, capturing the curvature of the road as it winds through rocky cliffs and lush green hills. The sunlight pierces through scattered clouds, reflecting off the car's rain-speckled surface, creating a dynamic, cinematic moment. The scene conveys a sense of freedom and exploration as the car disappears into the distance."
    second_prompt = "A woman with blonde hair styled up, wearing a black dress with sequins and pearl earrings, looks down with a sad expression on her face. The camera remains stationary, focused on the woman's face. The lighting is dim, casting soft shadows on her face. The scene appears to be from a movie or TV show."

    sample = {
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_attention_mask": None,
        "negative_prompt_attention_mask": None,
        "media_items": None,
    }

    def get_images(prompts):
        generators = [
            torch.Generator(device=device).manual_seed(params["seed"]) for _ in range(2)
        ]
        torch.manual_seed(params["seed"])

        images = pipeline(
            prompt=prompts,
            generator=generators,
            **sample,
            **params,
            pipeline_config=temp_config_path,
        ).images
        return images

    batch_diff_images = get_images([first_prompt, second_prompt])
    batch_same_images = get_images([second_prompt, second_prompt])

    # Take the second image from both runs
    image2_not_same = batch_diff_images[1, :, 0, :, :]
    image2_same = batch_same_images[1, :, 0, :, :]

    # Compute mean absolute difference, should be 0
    mad = torch.mean(torch.abs(image2_not_same - image2_same)).item()
    print(f"Mean absolute difference: {mad}")

    assert torch.allclose(image2_not_same, image2_same)


def test_prompt_enhancement(tmp_path, test_paths, monkeypatch):
    # Create pipeline with prompt enhancement enabled
    device = get_device()
    pipeline = create_ltx_video_pipeline(
        ckpt_path=test_paths["ckpt_path"],
        device=device,
        precision="bfloat16",
        text_encoder_model_name_or_path=test_paths["text_encoder_model_name_or_path"],
        enhance_prompt=True,
        prompt_enhancer_image_caption_model_name_or_path=test_paths[
            "prompt_enhancer_image_caption_model_name_or_path"
        ],
        prompt_enhancer_llm_model_name_or_path=test_paths[
            "prompt_enhancer_llm_model_name_or_path"
        ],
    )

    original_prompt = "A cat sitting on a windowsill"

    # Mock the pipeline's _encode_prompt method to verify the prompt being used
    original_encode_prompt = pipeline.encode_prompt

    prompts_used = []

    def mock_encode_prompt(prompt, *args, **kwargs):
        prompts_used.append(prompt[0] if isinstance(prompt, list) else prompt)
        return original_encode_prompt(prompt, *args, **kwargs)

    pipeline.encode_prompt = mock_encode_prompt

    # Set up minimal parameters for a quick test
    params = {
        "seed": 42,
        "image_cond_noise_scale": 0.15,
        "height": 512,
        "width": 768,
        "skip_layer_strategy": SkipLayerStrategy.AttentionValues,
        "num_frames": 1,
        "frame_rate": 25,
        "offload_to_cpu": False,
        "output_type": "pt",
        "is_video": False,
        "vae_per_channel_normalize": True,
        "mixed_precision": False,
    }

    config = {
        "pipeline_type": "base",
        "num_inference_steps": 1,
        "guidance_scale": 2.5,
        "stg_scale": 1,
        "rescaling_scale": 0.7,
        "skip_block_list": [1, 2],
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
    }

    temp_config_path = tmp_path / "config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    # Run pipeline with prompt enhancement enabled
    _ = pipeline(
        prompt=original_prompt,
        negative_prompt="worst quality",
        enhance_prompt=True,
        **params,
        pipeline_config=temp_config_path,
    )

    # Verify that the enhanced prompt was used
    assert len(prompts_used) > 0
    assert (
        prompts_used[0] != original_prompt
    ), f"Expected enhanced prompt to be different from original prompt, but got: {original_prompt}"

    # Run pipeline with prompt enhancement disabled
    prompts_used.clear()
    _ = pipeline(
        prompt=original_prompt,
        negative_prompt="worst quality",
        enhance_prompt=False,
        **params,
        pipeline_config=temp_config_path,
    )

    # Verify that the original prompt was used
    assert len(prompts_used) > 0
    assert (
        prompts_used[0] == original_prompt
    ), f"Expected original prompt to be used, but got: {prompts_used[0]}"


@pytest.mark.parametrize(
    "output_filename, output_format, expected_name, expected_ext",
    [
        (None, None, "video_output_0_a-young-woman_42_512x768x25_0", ".mp4"), # Default case
        ("custom_video", None, "custom_video", ".mp4"), # Custom name, default format
        ("custom_video.avi", None, "custom_video", ".avi"), # Custom name with ext, default format
        (None, "gif", "video_output_0_a-young-woman_42_512x768x25_0", ".gif"), # Default name, gif format
        ("custom_video", "gif", "custom_video", ".gif"), # Custom name, gif format
        ("custom_video.mp4", "gif", "custom_video", ".gif"), # Custom name with ext, gif format (format arg takes precedence)
        ("custom_image", None, "custom_image", ".png"), # Custom name for single image
        ("custom_image.jpeg", None, "custom_image", ".jpeg"), # Custom name with ext for single image
    ],
    ids=lambda x: f"{x}"
)
def test_output_filename_and_format(tmp_path, test_paths, output_filename, output_format, expected_name, expected_ext):
    is_single_image_test = "image" in expected_name
    num_frames = 1 if is_single_image_test else 25 # 1 frame for image, 25 for video

    params = {
        "seed": 42,
        "image_cond_noise_scale": 0.15,
        "height": 512,
        "width": 768,
        "num_frames": num_frames,
        "frame_rate": 25,
        "prompt": "A young woman", # Shorter prompt for simpler filename
        "negative_prompt": "worst quality",
        "offload_to_cpu": False,
        "output_path_arg": tmp_path,
        "output_filename_arg": output_filename,
        "output_format_arg": output_format,
    }

    config_data = {
        "pipeline_type": "base",
        "num_inference_steps": 1, # Faster test
        "guidance_scale": 2.5,
        "stg_scale": 1,
        "stg_rescale": 0.7,
        "stg_mode": "attention_values",
        "stg_skip_layers": "1",
        "precision": "bfloat16",
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
        "checkpoint_path": test_paths["ckpt_path"],
        "text_encoder_model_name_or_path": test_paths["text_encoder_model_name_or_path"],
        "prompt_enhancer_image_caption_model_name_or_path": test_paths["prompt_enhancer_image_caption_model_name_or_path"],
        "prompt_enhancer_llm_model_name_or_path": test_paths["prompt_enhancer_llm_model_name_or_path"],
        "prompt_enhancement_words_threshold": 0, # Disable enhancement for predictable filename
        "sampler": "from_checkpoint",
    }

    temp_config_path = tmp_path / "config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config_data, f)

    # For single image, base name will be image_output_0...
    if is_single_image_test and output_filename is None:
        expected_name = "image_output_0_a-young-woman_42_512x768x1_0"


    infer(**{**params, "pipeline_config": temp_config_path})

    expected_file_path = tmp_path / (expected_name + expected_ext)
    assert expected_file_path.exists(), f"Expected output file {expected_file_path} was not created."

    # Verify file is not empty
    assert expected_file_path.stat().st_size > 0, f"File {expected_file_path} is empty."

    # Rudimentary check for format based on extension (more robust checks would require imageio/ffmpeg)
    if not is_single_image_test: # Video checks
        if expected_ext == ".gif":
            # Try to read as gif
            try:
                import imageio
                imageio.mimread(expected_file_path)
            except Exception as e:
                pytest.fail(f"Failed to read generated GIF {expected_file_path}: {e}")
        elif expected_ext == ".mp4":
            # Try to read as mp4
            try:
                import imageio
                reader = imageio.get_reader(expected_file_path)
                assert reader.count_frames() > 0
                reader.close()
            except Exception as e:
                pytest.fail(f"Failed to read generated MP4 {expected_file_path}: {e}")
    else: # Single image check
        try:
            from PIL import Image
            Image.open(expected_file_path)
        except Exception as e:
            pytest.fail(f"Failed to read generated image {expected_file_path}: {e}")
