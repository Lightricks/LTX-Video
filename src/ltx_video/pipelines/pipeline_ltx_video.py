# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py
import copy
import inspect
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    latent_to_pixel_coords,
    vae_decode,
    vae_encode,
)
from ltx_video.models.transformers.symmetric_patchifier import Patchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.schedulers.rf import TimestepShifter
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.utils.prompt_enhance_utils import generate_cinematic_prompt
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from ltx_video.models.autoencoders.vae_encode import (
    un_normalize_latents,
    normalize_latents,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_512_BIN = {
    "0.25": [256.0, 1024.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "3.0": [864.0, 288.0],
    "4.0": [1024.0, 256.0],
}


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    max_timestep: Optional[float] = 1.0,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
            timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
            must be `None`.
        max_timestep ('float', *optional*, defaults to 1.0):
            The initial noising level for image-to-image/video-to-video. The list if timestamps will be
            truncated to start with a timestamp greater or equal to this.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    if max_timestep < 1.0:
        if max_timestep < timesteps.min():
            raise ValueError(
                f"max_timestep {max_timestep} is smaller than the minimum timestep {timesteps.min()}"
            )
        timesteps = timesteps[timesteps <= max_timestep]
        num_inference_steps = len(timesteps)
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)

    return timesteps, num_inference_steps


@dataclass
class ConditioningItem:
    """
    Defines a single frame-conditioning item - a single frame or a sequence of frames.

    Attributes:
        media_item (torch.Tensor): shape=(b, 3, f, h, w). The media item to condition on.
        media_frame_number (int): The start-frame number of the media item in the generated video.
        conditioning_strength (float): The strength of the conditioning (1.0 = full conditioning).
        media_x (Optional[int]): Optional left x coordinate of the media item in the generated frame.
        media_y (Optional[int]): Optional top y coordinate of the media item in the generated frame.
    """

    media_item: torch.Tensor
    media_frame_number: int
    conditioning_strength: float
    media_x: Optional[int] = None
    media_y: Optional[int] = None


class LTXVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using LTX-Video.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. This uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`Transformer2DModel`]):
            A text conditioned `Transformer2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    _optional_components = [
        "tokenizer",
        "text_encoder",
        "prompt_enhancer_image_caption_model",
        "prompt_enhancer_image_caption_processor",
        "prompt_enhancer_llm_model",
        "prompt_enhancer_llm_tokenizer",
    ]
    model_cpu_offload_seq = "prompt_enhancer_image_caption_model->prompt_enhancer_llm_model->text_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: Transformer3DModel,
        scheduler: DPMSolverMultistepScheduler,
        patchifier: Patchifier,
        prompt_enhancer_image_caption_model: AutoModelForCausalLM,
        prompt_enhancer_image_caption_processor: AutoProcessor,
        prompt_enhancer_llm_model: AutoModelForCausalLM,
        prompt_enhancer_llm_tokenizer: AutoTokenizer,
        allowed_inference_steps: Optional[List[float]] = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            patchifier=patchifier,
            prompt_enhancer_image_caption_model=prompt_enhancer_image_caption_model,
            prompt_enhancer_image_caption_processor=prompt_enhancer_image_caption_processor,
            prompt_enhancer_llm_model=prompt_enhancer_llm_model,
            prompt_enhancer_llm_tokenizer=prompt_enhancer_llm_tokenizer,
        )

        self.video_scale_factor, self.vae_scale_factor, _ = get_vae_size_scale_factor(
            self.vae
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.allowed_inference_steps = allowed_inference_steps

    def mask_text_embeddings(self, emb, mask):
        if emb.shape[0] == 1:
            keep_index = mask.sum().item()
            return emb[:, :, :keep_index, :], keep_index
        else:
            masked_feature = emb * mask[:, None, :, None]
            return masked_feature, emb.shape[2]

    # Adapted from diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        text_encoder_max_tokens: int = 256,
        **kwargs,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                This should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings.
        """

        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # See Section 3.1. of the paper.
        max_length = (
            text_encoder_max_tokens  # TPU supports only lengths multiple of 128
        )
        if prompt_embeds is None:
            assert (
                self.text_encoder is not None
            ), "You should provide either prompt_embeds or self.text_encoder should not be None,"
            text_enc_device = next(self.text_encoder.parameters()).device
            prompt = self._text_preprocessing(prompt)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask
            prompt_attention_mask = prompt_attention_mask.to(text_enc_device)
            prompt_attention_mask = prompt_attention_mask.to(device)

            prompt_embeds = self.text_encoder(
                text_input_ids.to(text_enc_device), attention_mask=prompt_attention_mask
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        prompt_attention_mask = prompt_attention_mask.repeat(1, num_images_per_prompt)
        prompt_attention_mask = prompt_attention_mask.view(
            bs_embed * num_images_per_prompt, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = self._text_preprocessing(negative_prompt)
            uncond_tokens = uncond_tokens * batch_size
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_prompt_attention_mask = uncond_input.attention_mask
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                text_enc_device
            )

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(text_enc_device),
                attention_mask=negative_prompt_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(
                1, num_images_per_prompt
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                bs_embed * num_images_per_prompt, -1
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        enhance_prompt=False,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError(
                "Must provide `prompt_attention_mask` when specifying `prompt_embeds`."
            )

        if (
            negative_prompt_embeds is not None
            and negative_prompt_attention_mask is None
        ):
            raise ValueError(
                "Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

        if enhance_prompt:
            assert (
                self.prompt_enhancer_image_caption_model is not None
            ), "Image caption model must be initialized if enhance_prompt is True"
            assert (
                self.prompt_enhancer_image_caption_processor is not None
            ), "Image caption processor must be initialized if enhance_prompt is True"
            assert (
                self.prompt_enhancer_llm_model is not None
            ), "Text prompt enhancer model must be initialized if enhance_prompt is True"
            assert (
                self.prompt_enhancer_llm_tokenizer is not None
            ), "Text prompt enhancer tokenizer must be initialized if enhance_prompt is True"

    def _text_preprocessing(self, text):
        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            text = text.strip()
            return text

        return [process(t) for t in text]

    @staticmethod
    def add_noise_to_image_conditioning_latents(
        t: float,
        init_latents: torch.Tensor,
        latents: torch.Tensor,
        noise_scale: float,
        conditioning_mask: torch.Tensor,
        generator,
        eps=1e-6,
    ):
        """
        Add timestep-dependent noise to the hard-conditioning latents.
        This helps with motion continuity, especially when conditioned on a single frame.
        """
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        # Add noise only to hard-conditioning latents (conditioning_mask = 1.0)
        need_to_noise = (conditioning_mask > 1.0 - eps).unsqueeze(-1)
        noised_latents = init_latents + noise_scale * noise * (t**2)
        latents = torch.where(need_to_noise, noised_latents, latents)
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self,
        latents: torch.Tensor | None,
        media_items: torch.Tensor | None,
        timestep: float,
        latent_shape: torch.Size | Tuple[Any, ...],
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | List[torch.Generator],
        vae_per_channel_normalize: bool = True,
    ):
        """
        Prepare the initial latent tensor to be denoised.
        The latents are either pure noise or a noised version of the encoded media items.
        Args:
            latents (`torch.FloatTensor` or `None`):
                The latents to use (provided by the user) or `None` to create new latents.
            media_items (`torch.FloatTensor` or `None`):
                An image or video to be updated using img2img or vid2vid. The media item is encoded and noised.
            timestep (`float`):
                The timestep to noise the encoded media_items to.
            latent_shape (`torch.Size`):
                The target latent shape.
            dtype (`torch.dtype`):
                The target dtype.
            device (`torch.device`):
                The target device.
            generator (`torch.Generator` or `List[torch.Generator]`):
                Generator(s) to be used for the noising process.
            vae_per_channel_normalize ('bool'):
                When encoding the media_items, whether to normalize the latents per-channel.
        Returns:
            `torch.FloatTensor`: The latents to be used for the denoising process. This is a tensor of shape
            (batch_size, num_channels, height, width).
        """
        if isinstance(generator, list) and len(generator) != latent_shape[0]:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {latent_shape[0]}. Make sure the batch size matches the length of the generators."
            )

        # Initialize the latents with the given latents or encoded media item, if provided
        assert (
            latents is None or media_items is None
        ), "Cannot provide both latents and media_items. Please provide only one of the two."

        assert (
            latents is None and media_items is None or timestep < 1.0
        ), "Input media_item or latents are provided, but they will be replaced with noise."

        if media_items is not None:
            latents = vae_encode(
                media_items.to(dtype=self.vae.dtype, device=self.vae.device),
                self.vae,
                vae_per_channel_normalize=vae_per_channel_normalize,
            )
        if latents is not None:
            assert (
                latents.shape == latent_shape
            ), f"Latents have to be of shape {latent_shape} but are {latents.shape}."
            latents = latents.to(device=device, dtype=dtype)

        # For backward compatibility, generate in the "patchified" shape and rearrange
        b, c, f, h, w = latent_shape
        noise = randn_tensor(
            (b, f * h * w, c), generator=generator, device=device, dtype=dtype
        )
        noise = rearrange(noise, "b (f h w) c -> b c f h w", f=f, h=h, w=w)

        # scale the initial noise by the standard deviation required by the scheduler
        noise = noise * self.scheduler.init_noise_sigma

        if latents is None:
            latents = noise
        else:
            # Noise the latents to the required (first) timestep
            latents = timestep * noise + (1 - timestep) * latents

        return latents

    @staticmethod
    def classify_height_width_bin(
        height: int, width: int, ratios: dict
    ) -> Tuple[int, int]:
        """Returns binned height and width."""
        ar = float(height / width)
        closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
        default_hw = ratios[closest_ratio]
        return int(default_hw[0]), int(default_hw[1])

    @staticmethod
    def resize_and_crop_tensor(
        samples: torch.Tensor, new_width: int, new_height: int
    ) -> torch.Tensor:
        n_frames, orig_height, orig_width = samples.shape[-3:]

        # Check if resizing is needed
        if orig_height != new_height or orig_width != new_width:
            ratio = max(new_height / orig_height, new_width / orig_width)
            resized_width = int(orig_width * ratio)
            resized_height = int(orig_height * ratio)

            # Resize
            samples = LTXVideoPipeline.resize_tensor(
                samples, resized_height, resized_width
            )

            # Center Crop
            start_x = (resized_width - new_width) // 2
            end_x = start_x + new_width
            start_y = (resized_height - new_height) // 2
            end_y = start_y + new_height
            samples = samples[..., start_y:end_y, start_x:end_x]

        return samples

    @staticmethod
    def resize_tensor(media_items, height, width):
        n_frames = media_items.shape[2]
        if media_items.shape[-2:] != (height, width):
            media_items = rearrange(media_items, "b c n h w -> (b n) c h w")
            media_items = F.interpolate(
                media_items,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            media_items = rearrange(media_items, "(b n) c h w -> b c n h w", n=n_frames)
        return media_items

    @torch.no_grad()
    def __call__(
        self,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: Union[float, List[float]] = 4.5,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        skip_block_list: Optional[Union[List[List[int]], List[int]]] = None,
        stg_scale: Union[float, List[float]] = 1.0,
        rescaling_scale: Union[float, List[float]] = 0.7,
        guidance_timesteps: Optional[List[int]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        conditioning_items: Optional[List[ConditioningItem]] = None,
        decode_timestep: Union[List[float], float] = 0.0,
        decode_noise_scale: Optional[List[float]] = None,
        mixed_precision: bool = False,
        offload_to_cpu: bool = False,
        enhance_prompt: bool = False,
        text_encoder_max_tokens: int = 256,
        stochastic_sampling: bool = False,
        media_items: Optional[torch.Tensor] = None,
        strength: Optional[float] = 1.0,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. If `timesteps` is provided, this parameter is ignored.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.FloatTensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. This negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            enhance_prompt (`bool`, *optional*, defaults to `False`):
                If set to `True`, the prompt is enhanced using a LLM model.
            text_encoder_max_tokens (`int`, *optional*, defaults to `256`):
                The maximum number of tokens to use for the text encoder.
            stochastic_sampling (`bool`, *optional*, defaults to `False`):
                If set to `True`, the sampling is stochastic. If set to `False`, the sampling is deterministic.
            media_items ('torch.Tensor', *optional*):
                The input media item used for image-to-image / video-to-video.
                When provided, they will be noised according to 'strength' and then fully denoised.
            strength ('floaty', *optional* defaults to 1.0):
                The editing level in image-to-image / video-to-video. The provided input will be noised
                to this level.
        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        is_video = kwargs.get("is_video", False)
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        self.video_scale_factor = self.video_scale_factor if is_video else 1
        vae_per_channel_normalize = kwargs.get("vae_per_channel_normalize", True)
        image_cond_noise_scale = kwargs.get("image_cond_noise_scale", 0.0)

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_num_frames = num_frames // self.video_scale_factor
        if isinstance(self.vae, CausalVideoAutoencoder) and is_video:
            latent_num_frames += 1
        latent_shape = (
            batch_size * num_images_per_prompt,
            self.transformer.config.in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
        )

        # Prepare the list of denoising time-steps

        retrieve_timesteps_kwargs = {}
        if isinstance(self.scheduler, TimestepShifter):
            retrieve_timesteps_kwargs["samples_shape"] = latent_shape

        assert strength == 1.0 or latents is not None or media_items is not None, (
            "strength < 1 is used for image-to-image/video-to-video - "
            "media_item or latents should be provided."
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            max_timestep=strength,
            **retrieve_timesteps_kwargs,
        )
        if self.allowed_inference_steps is not None:
            for timestep in [round(x, 4) for x in timesteps.tolist()]:
                assert (
                    timestep in self.allowed_inference_steps
                ), f"Invalid inference timestep {timestep}. Allowed timesteps are {self.allowed_inference_steps}."

        if guidance_timesteps:
            guidance_mapping = []
            for timestep in timesteps:
                indices = [
                    i for i, val in enumerate(guidance_timesteps) if val <= timestep
                ]
                # assert len(indices) > 0, f"No guidance timestep found for {timestep}"
                guidance_mapping.append(
                    indices[0] if len(indices) > 0 else (len(guidance_timesteps) - 1)
                )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        if not isinstance(guidance_scale, List):
            guidance_scale = [guidance_scale] * len(timesteps)
        else:
            guidance_scale = [
                guidance_scale[guidance_mapping[i]] for i in range(len(timesteps))
            ]

        # For simplicity, we are using a constant num_conds for all timesteps, so we need to zero
        # out cases where the guidance scale should not be applied.
        guidance_scale = [x if x > 1.0 else 0.0 for x in guidance_scale]

        if not isinstance(stg_scale, List):
            stg_scale = [stg_scale] * len(timesteps)
        else:
            stg_scale = [stg_scale[guidance_mapping[i]] for i in range(len(timesteps))]

        if not isinstance(rescaling_scale, List):
            rescaling_scale = [rescaling_scale] * len(timesteps)
        else:
            rescaling_scale = [
                rescaling_scale[guidance_mapping[i]] for i in range(len(timesteps))
            ]

        do_classifier_free_guidance = any(x > 1.0 for x in guidance_scale)
        do_spatio_temporal_guidance = any(x > 0.0 for x in stg_scale)
        do_rescaling = any(x != 1.0 for x in rescaling_scale)

        num_conds = 1
        if do_classifier_free_guidance:
            num_conds += 1
        if do_spatio_temporal_guidance:
            num_conds += 1

        # Normalize skip_block_list to always be None or a list of lists matching timesteps
        if skip_block_list is not None:
            # Convert single list to list of lists if needed
            if not isinstance(skip_block_list[0], list):
                skip_block_list = [skip_block_list] * len(timesteps)
            else:
                new_skip_block_list = []
                for i, timestep in enumerate(timesteps):
                    new_skip_block_list.append(skip_block_list[guidance_mapping[i]])
                skip_block_list = new_skip_block_list

        # Prepare skip layer masks
        skip_layer_masks: Optional[List[torch.Tensor]] = None
        if do_spatio_temporal_guidance:
            if skip_block_list is not None:
                skip_layer_masks = [
                    self.transformer.create_skip_layer_mask(
                        batch_size, num_conds, num_conds - 1, skip_blocks
                    )
                    for skip_blocks in skip_block_list
                ]

        if enhance_prompt:
            self.prompt_enhancer_image_caption_model = (
                self.prompt_enhancer_image_caption_model.to(self._execution_device)
            )
            self.prompt_enhancer_llm_model = self.prompt_enhancer_llm_model.to(
                self._execution_device
            )

            prompt = generate_cinematic_prompt(
                self.prompt_enhancer_image_caption_model,
                self.prompt_enhancer_image_caption_processor,
                self.prompt_enhancer_llm_model,
                self.prompt_enhancer_llm_tokenizer,
                prompt,
                conditioning_items,
                max_new_tokens=text_encoder_max_tokens,
            )

        # 3. Encode input prompt
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self._execution_device)

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            text_encoder_max_tokens=text_encoder_max_tokens,
        )

        if offload_to_cpu and self.text_encoder is not None:
            self.text_encoder = self.text_encoder.cpu()

        self.transformer = self.transformer.to(self._execution_device)

        prompt_embeds_batch = prompt_embeds
        prompt_attention_mask_batch = prompt_attention_mask
        if do_classifier_free_guidance:
            prompt_embeds_batch = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0
            )
            prompt_attention_mask_batch = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )
        if do_spatio_temporal_guidance:
            prompt_embeds_batch = torch.cat([prompt_embeds_batch, prompt_embeds], dim=0)
            prompt_attention_mask_batch = torch.cat(
                [
                    prompt_attention_mask_batch,
                    prompt_attention_mask,
                ],
                dim=0,
            )

        # 4. Prepare the initial latents using the provided media and conditioning items

        # Prepare the initial latents tensor, shape = (b, c, f, h, w)
        latents = self.prepare_latents(
            latents=latents,
            media_items=media_items,
            timestep=timesteps[0],
            latent_shape=latent_shape,
            dtype=prompt_embeds_batch.dtype,
            device=device,
            generator=generator,
            vae_per_channel_normalize=vae_per_channel_normalize,
        )

        # Update the latents with the conditioning items and patchify them into (b, n, c)
        latents, pixel_coords, conditioning_mask, num_cond_latents = (
            self.prepare_conditioning(
                conditioning_items=conditioning_items,
                init_latents=latents,
                num_frames=num_frames,
                height=height,
                width=width,
                vae_per_channel_normalize=vae_per_channel_normalize,
                generator=generator,
            )
        )
        init_latents = latents.clone()  # Used for image_cond_noise_update

        pixel_coords = torch.cat([pixel_coords] * num_conds)
        orig_conditioning_mask = conditioning_mask
        if conditioning_mask is not None and is_video:
            assert num_images_per_prompt == 1
            conditioning_mask = torch.cat([conditioning_mask] * num_conds)
        fractional_coords = pixel_coords.to(torch.float32)
        fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if conditioning_mask is not None and image_cond_noise_scale > 0.0:
                    latents = self.add_noise_to_image_conditioning_latents(
                        t,
                        init_latents,
                        latents,
                        image_cond_noise_scale,
                        orig_conditioning_mask,
                        generator,
                    )

                latent_model_input = (
                    torch.cat([latents] * num_conds) if num_conds > 1 else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor(
                        [current_timestep],
                        dtype=dtype,
                        device=latent_model_input.device,
                    )
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(
                        latent_model_input.device
                    )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(
                    latent_model_input.shape[0]
                ).unsqueeze(-1)

                if conditioning_mask is not None:
                    # Conditioning latents have an initial timestep and noising level of (1.0 - conditioning_mask)
                    # and will start to be denoised when the current timestep is lower than their conditioning timestep.
                    current_timestep = torch.min(
                        current_timestep, 1.0 - conditioning_mask
                    )

                # Choose the appropriate context manager based on `mixed_precision`
                if mixed_precision:
                    context_manager = torch.autocast(device.type, dtype=torch.bfloat16)
                else:
                    context_manager = nullcontext()  # Dummy context manager

                # predict noise model_output
                with context_manager:
                    noise_pred = self.transformer(
                        latent_model_input.to(self.transformer.dtype),
                        indices_grid=fractional_coords,
                        encoder_hidden_states=prompt_embeds_batch.to(
                            self.transformer.dtype
                        ),
                        encoder_attention_mask=prompt_attention_mask_batch,
                        timestep=current_timestep,
                        skip_layer_mask=(
                            skip_layer_masks[i]
                            if skip_layer_masks is not None
                            else None
                        ),
                        skip_layer_strategy=skip_layer_strategy,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_spatio_temporal_guidance:
                    noise_pred_text, noise_pred_text_perturb = noise_pred.chunk(
                        num_conds
                    )[-2:]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_conds)[:2]
                    noise_pred = noise_pred_uncond + guidance_scale[i] * (
                        noise_pred_text - noise_pred_uncond
                    )
                elif do_spatio_temporal_guidance:
                    noise_pred = noise_pred_text
                if do_spatio_temporal_guidance:
                    noise_pred = noise_pred + stg_scale[i] * (
                        noise_pred_text - noise_pred_text_perturb
                    )
                    if do_rescaling and stg_scale[i] > 0.0:
                        noise_pred_text_std = noise_pred_text.view(batch_size, -1).std(
                            dim=1, keepdim=True
                        )
                        noise_pred_std = noise_pred.view(batch_size, -1).std(
                            dim=1, keepdim=True
                        )

                        factor = noise_pred_text_std / noise_pred_std
                        factor = rescaling_scale[i] * factor + (1 - rescaling_scale[i])

                        noise_pred = noise_pred * factor.view(batch_size, 1, 1)

                current_timestep = current_timestep[:1]
                # learned sigma
                if (
                    self.transformer.config.out_channels // 2
                    == self.transformer.config.in_channels
                ):
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # compute previous image: x_t -> x_t-1
                latents = self.denoising_step(
                    latents,
                    noise_pred,
                    current_timestep,
                    orig_conditioning_mask,
                    t,
                    extra_step_kwargs,
                    stochastic_sampling=stochastic_sampling,
                )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if callback_on_step_end is not None:
                    callback_on_step_end(self, i, t, {})

        if offload_to_cpu:
            self.transformer = self.transformer.cpu()
            if self._execution_device == "cuda":
                torch.cuda.empty_cache()

        # Remove the added conditioning latents
        latents = latents[:, num_cond_latents:]

        latents = self.patchifier.unpatchify(
            latents=latents,
            output_height=latent_height,
            output_width=latent_width,
            out_channels=self.transformer.in_channels
            // math.prod(self.patchifier.patch_size),
        )
        if output_type != "latent":
            if self.vae.decoder.timestep_conditioning:
                noise = torch.randn_like(latents)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * latents.shape[0]
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * latents.shape[0]

                decode_timestep = torch.tensor(decode_timestep).to(latents.device)
                decode_noise_scale = torch.tensor(decode_noise_scale).to(
                    latents.device
                )[:, None, None, None, None]
                latents = (
                    latents * (1 - decode_noise_scale) + noise * decode_noise_scale
                )
            else:
                decode_timestep = None
            image = vae_decode(
                latents,
                self.vae,
                is_video,
                vae_per_channel_normalize=kwargs["vae_per_channel_normalize"],
                timestep=decode_timestep,
            )

            image = self.image_processor.postprocess(image, output_type=output_type)

        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def denoising_step(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        current_timestep: torch.Tensor,
        conditioning_mask: torch.Tensor,
        t: float,
        extra_step_kwargs,
        t_eps=1e-6,
        stochastic_sampling=False,
    ):
        """
        Perform the denoising step for the required tokens, based on the current timestep and
        conditioning mask:
        Conditioning latents have an initial timestep and noising level of (1.0 - conditioning_mask)
        and will start to be denoised when the current timestep is equal or lower than their
        conditioning timestep.
        (hard-conditioning latents with conditioning_mask = 1.0 are never denoised)
        """
        # Denoise the latents using the scheduler
        denoised_latents = self.scheduler.step(
            noise_pred,
            t if current_timestep is None else current_timestep,
            latents,
            **extra_step_kwargs,
            return_dict=False,
            stochastic_sampling=stochastic_sampling,
        )[0]

        if conditioning_mask is None:
            return denoised_latents

        tokens_to_denoise_mask = (t - t_eps < (1.0 - conditioning_mask)).unsqueeze(-1)
        return torch.where(tokens_to_denoise_mask, denoised_latents, latents)

    def prepare_conditioning(
        self,
        conditioning_items: Optional[List[ConditioningItem]],
        init_latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        vae_per_channel_normalize: bool = False,
        generator=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Prepare conditioning tokens based on the provided conditioning items.

        This method encodes provided conditioning items (video frames or single frames) into latents
        and integrates them with the initial latent tensor. It also calculates corresponding pixel
        coordinates, a mask indicating the influence of conditioning latents, and the total number of
        conditioning latents.

        Args:
            conditioning_items (Optional[List[ConditioningItem]]): A list of ConditioningItem objects.
            init_latents (torch.Tensor): The initial latent tensor of shape (b, c, f_l, h_l, w_l), where
                `f_l` is the number of latent frames, and `h_l` and `w_l` are latent spatial dimensions.
            num_frames, height, width: The dimensions of the generated video.
            vae_per_channel_normalize (bool, optional): Whether to normalize channels during VAE encoding.
                Defaults to `False`.
            generator: The random generator

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
                - `init_latents` (torch.Tensor): The updated latent tensor including conditioning latents,
                  patchified into (b, n, c) shape.
                - `init_pixel_coords` (torch.Tensor): The pixel coordinates corresponding to the updated
                  latent tensor.
                - `conditioning_mask` (torch.Tensor): A mask indicating the conditioning-strength of each
                  latent token.
                - `num_cond_latents` (int): The total number of latent tokens added from conditioning items.

        Raises:
            AssertionError: If input shapes, dimensions, or conditions for applying conditioning are invalid.
        """
        assert isinstance(self.vae, CausalVideoAutoencoder)

        if conditioning_items:
            batch_size, _, num_latent_frames = init_latents.shape[:3]

            init_conditioning_mask = torch.zeros(
                init_latents[:, 0, :, :, :].shape,
                dtype=torch.float32,
                device=init_latents.device,
            )

            extra_conditioning_latents = []
            extra_conditioning_pixel_coords = []
            extra_conditioning_mask = []
            extra_conditioning_num_latents = 0  # Number of extra conditioning latents added (should be removed before decoding)

            # Process each conditioning item
            for conditioning_item in conditioning_items:
                conditioning_item = self._resize_conditioning_item(
                    conditioning_item, height, width
                )
                media_item = conditioning_item.media_item
                media_frame_number = conditioning_item.media_frame_number
                strength = conditioning_item.conditioning_strength
                assert media_item.ndim == 5  # (b, c, f, h, w)
                b, c, n_frames, h, w = media_item.shape
                assert (
                    height == h and width == w
                ) or media_frame_number == 0, f"Dimensions do not match: {height}x{width} != {h}x{w} - allowed only when media_frame_number == 0"
                assert n_frames % 8 == 1
                assert (
                    media_frame_number >= 0
                    and media_frame_number + n_frames <= num_frames
                )

                # Encode the provided conditioning media item
                media_item_latents = vae_encode(
                    media_item.to(dtype=self.vae.dtype, device=self.vae.device),
                    self.vae,
                    vae_per_channel_normalize=vae_per_channel_normalize,
                ).to(dtype=init_latents.dtype)

                # Handle the different conditioning cases
                if media_frame_number == 0:
                    # Get the target spatial position of the latent conditioning item
                    media_item_latents, l_x, l_y = self._get_latent_spatial_position(
                        media_item_latents,
                        conditioning_item,
                        height,
                        width,
                        strip_latent_border=True,
                    )
                    b, c_l, f_l, h_l, w_l = media_item_latents.shape

                    # First frame or sequence - just update the initial noise latents and the mask
                    init_latents[:, :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l] = (
                        torch.lerp(
                            init_latents[:, :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l],
                            media_item_latents,
                            strength,
                        )
                    )
                    init_conditioning_mask[
                        :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l
                    ] = strength
                else:
                    # Non-first frame or sequence
                    if n_frames > 1:
                        # Handle non-first sequence.
                        # Encoded latents are either fully consumed, or the prefix is handled separately below.
                        (
                            init_latents,
                            init_conditioning_mask,
                            media_item_latents,
                        ) = self._handle_non_first_conditioning_sequence(
                            init_latents,
                            init_conditioning_mask,
                            media_item_latents,
                            media_frame_number,
                            strength,
                        )

                    # Single frame or sequence-prefix latents
                    if media_item_latents is not None:
                        noise = randn_tensor(
                            media_item_latents.shape,
                            generator=generator,
                            device=media_item_latents.device,
                            dtype=media_item_latents.dtype,
                        )

                        media_item_latents = torch.lerp(
                            noise, media_item_latents, strength
                        )

                        # Patchify the extra conditioning latents and calculate their pixel coordinates
                        media_item_latents, latent_coords = self.patchifier.patchify(
                            latents=media_item_latents
                        )
                        pixel_coords = latent_to_pixel_coords(
                            latent_coords,
                            self.vae,
                            causal_fix=self.transformer.config.causal_temporal_positioning,
                        )

                        # Update the frame numbers to match the target frame number
                        pixel_coords[:, 0] += media_frame_number
                        extra_conditioning_num_latents += media_item_latents.shape[1]

                        conditioning_mask = torch.full(
                            media_item_latents.shape[:2],
                            strength,
                            dtype=torch.float32,
                            device=init_latents.device,
                        )

                        extra_conditioning_latents.append(media_item_latents)
                        extra_conditioning_pixel_coords.append(pixel_coords)
                        extra_conditioning_mask.append(conditioning_mask)

        # Patchify the updated latents and calculate their pixel coordinates
        init_latents, init_latent_coords = self.patchifier.patchify(
            latents=init_latents
        )
        init_pixel_coords = latent_to_pixel_coords(
            init_latent_coords,
            self.vae,
            causal_fix=self.transformer.config.causal_temporal_positioning,
        )

        if not conditioning_items:
            return init_latents, init_pixel_coords, None, 0

        init_conditioning_mask, _ = self.patchifier.patchify(
            latents=init_conditioning_mask.unsqueeze(1)
        )
        init_conditioning_mask = init_conditioning_mask.squeeze(-1)

        if extra_conditioning_latents:
            # Stack the extra conditioning latents, pixel coordinates and mask
            init_latents = torch.cat([*extra_conditioning_latents, init_latents], dim=1)
            init_pixel_coords = torch.cat(
                [*extra_conditioning_pixel_coords, init_pixel_coords], dim=2
            )
            init_conditioning_mask = torch.cat(
                [*extra_conditioning_mask, init_conditioning_mask], dim=1
            )

            if self.transformer.use_tpu_flash_attention:
                # When flash attention is used, keep the original number of tokens by removing
                #   tokens from the end.
                init_latents = init_latents[:, :-extra_conditioning_num_latents]
                init_pixel_coords = init_pixel_coords[
                    :, :, :-extra_conditioning_num_latents
                ]
                init_conditioning_mask = init_conditioning_mask[
                    :, :-extra_conditioning_num_latents
                ]

        return (
            init_latents,
            init_pixel_coords,
            init_conditioning_mask,
            extra_conditioning_num_latents,
        )

    @staticmethod
    def _resize_conditioning_item(
        conditioning_item: ConditioningItem,
        height: int,
        width: int,
    ):
        if conditioning_item.media_x or conditioning_item.media_y:
            raise ValueError(
                "Provide media_item in the target size for spatial conditioning."
            )
        new_conditioning_item = copy.copy(conditioning_item)
        new_conditioning_item.media_item = LTXVideoPipeline.resize_tensor(
            conditioning_item.media_item, height, width
        )
        return new_conditioning_item

    def _get_latent_spatial_position(
        self,
        latents: torch.Tensor,
        conditioning_item: ConditioningItem,
        height: int,
        width: int,
        strip_latent_border,
    ):
        """
        Get the spatial position of the conditioning item in the latent space.
        If requested, strip the conditioning latent borders that do not align with target borders.
        (border latents look different then other latents and might confuse the model)
        """
        scale = self.vae_scale_factor
        h, w = conditioning_item.media_item.shape[-2:]
        assert (
            h <= height and w <= width
        ), f"Conditioning item size {h}x{w} is larger than target size {height}x{width}"
        assert h % scale == 0 and w % scale == 0

        # Compute the start and end spatial positions of the media item
        x_start, y_start = conditioning_item.media_x, conditioning_item.media_y
        x_start = (width - w) // 2 if x_start is None else x_start
        y_start = (height - h) // 2 if y_start is None else y_start
        x_end, y_end = x_start + w, y_start + h
        assert (
            x_end <= width and y_end <= height
        ), f"Conditioning item {x_start}:{x_end}x{y_start}:{y_end} is out of bounds for target size {width}x{height}"

        if strip_latent_border:
            # Strip one latent from left/right and/or top/bottom, update x, y accordingly
            if x_start > 0:
                x_start += scale
                latents = latents[:, :, :, :, 1:]

            if y_start > 0:
                y_start += scale
                latents = latents[:, :, :, 1:, :]

            if x_end < width:
                latents = latents[:, :, :, :, :-1]

            if y_end < height:
                latents = latents[:, :, :, :-1, :]

        return latents, x_start // scale, y_start // scale

    @staticmethod
    def _handle_non_first_conditioning_sequence(
        init_latents: torch.Tensor,
        init_conditioning_mask: torch.Tensor,
        latents: torch.Tensor,
        media_frame_number: int,
        strength: float,
        num_prefix_latent_frames: int = 2,
        prefix_latents_mode: str = "concat",
        prefix_soft_conditioning_strength: float = 0.15,
    ):
        """
        Special handling for a conditioning sequence that does not start on the first frame.
        The special handling is required to allow a short encoded video to be used as middle
        (or last) sequence in a longer video.
        Args:
            init_latents (torch.Tensor): The initial noise latents to be updated.
            init_conditioning_mask (torch.Tensor): The initial conditioning mask to be updated.
            latents (torch.Tensor): The encoded conditioning item.
            media_frame_number (int): The target frame number of the first frame in the conditioning sequence.
            strength (float): The conditioning strength for the conditioning latents.
            num_prefix_latent_frames (int, optional): The length of the sequence prefix, to be handled
                separately. Defaults to 2.
            prefix_latents_mode (str, optional): Special treatment for prefix (boundary) latents.
                - "drop": Drop the prefix latents.
                - "soft": Use the prefix latents, but with soft-conditioning
                - "concat": Add the prefix latents as extra tokens (like single frames)
            prefix_soft_conditioning_strength (float, optional): The strength of the soft-conditioning for
                the prefix latents, relevant if `prefix_latents_mode` is "soft". Defaults to 0.1.

        """
        f_l = latents.shape[2]
        f_l_p = num_prefix_latent_frames
        assert f_l >= f_l_p
        assert media_frame_number % 8 == 0
        if f_l > f_l_p:
            # Insert the conditioning latents **excluding the prefix** into the sequence
            f_l_start = media_frame_number // 8 + f_l_p
            f_l_end = f_l_start + f_l - f_l_p
            init_latents[:, :, f_l_start:f_l_end] = torch.lerp(
                init_latents[:, :, f_l_start:f_l_end],
                latents[:, :, f_l_p:],
                strength,
            )
            # Mark these latent frames as conditioning latents
            init_conditioning_mask[:, f_l_start:f_l_end] = strength

        # Handle the prefix-latents
        if prefix_latents_mode == "soft":
            if f_l_p > 1:
                # Drop the first (single-frame) latent and soft-condition the remaining prefix
                f_l_start = media_frame_number // 8 + 1
                f_l_end = f_l_start + f_l_p - 1
                strength = min(prefix_soft_conditioning_strength, strength)
                init_latents[:, :, f_l_start:f_l_end] = torch.lerp(
                    init_latents[:, :, f_l_start:f_l_end],
                    latents[:, :, 1:f_l_p],
                    strength,
                )
                # Mark these latent frames as conditioning latents
                init_conditioning_mask[:, f_l_start:f_l_end] = strength
            latents = None  # No more latents to handle
        elif prefix_latents_mode == "drop":
            # Drop the prefix latents
            latents = None
        elif prefix_latents_mode == "concat":
            # Pass-on the prefix latents to be handled as extra conditioning frames
            latents = latents[:, :, :f_l_p]
        else:
            raise ValueError(f"Invalid prefix_latents_mode: {prefix_latents_mode}")
        return (
            init_latents,
            init_conditioning_mask,
            latents,
        )

    def trim_conditioning_sequence(
        self, start_frame: int, sequence_num_frames: int, target_num_frames: int
    ):
        """
        Trim a conditioning sequence to the allowed number of frames.

        Args:
            start_frame (int): The target frame number of the first frame in the sequence.
            sequence_num_frames (int): The number of frames in the sequence.
            target_num_frames (int): The target number of frames in the generated video.

        Returns:
            int: updated sequence length
        """
        scale_factor = self.video_scale_factor
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        return num_frames


class LTXMultiScalePipeline:
    def _upsample_latents(
        self, latest_upsampler: LatentUpsampler, latents: torch.Tensor
    ):
        assert latents.device == latest_upsampler.device

        latents = un_normalize_latents(
            latents, self.vae, vae_per_channel_normalize=True
        )
        upsampled_latents = latest_upsampler(latents)
        upsampled_latents = normalize_latents(
            upsampled_latents, self.vae, vae_per_channel_normalize=True
        )
        return upsampled_latents

    def __init__(
        self, video_pipeline: LTXVideoPipeline, latent_upsampler: LatentUpsampler
    ):
        self.video_pipeline = video_pipeline
        self.vae = video_pipeline.vae
        self.latent_upsampler = latent_upsampler

    def __call__(
        self,
        downscale_factor: float,
        first_pass: dict,
        second_pass: dict,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        original_kwargs = kwargs.copy()
        original_output_type = kwargs["output_type"]
        original_width = kwargs["width"]
        original_height = kwargs["height"]

        x_width = int(kwargs["width"] * downscale_factor)
        downscaled_width = x_width - (x_width % self.video_pipeline.vae_scale_factor)
        x_height = int(kwargs["height"] * downscale_factor)
        downscaled_height = x_height - (x_height % self.video_pipeline.vae_scale_factor)

        kwargs["output_type"] = "latent"
        kwargs["width"] = downscaled_width
        kwargs["height"] = downscaled_height
        kwargs.update(**first_pass)
        result = self.video_pipeline(*args, **kwargs)
        latents = result.images

        upsampled_latents = self._upsample_latents(self.latent_upsampler, latents)

        kwargs = original_kwargs

        kwargs["latents"] = upsampled_latents
        kwargs["output_type"] = original_output_type
        kwargs["width"] = downscaled_width * 2
        kwargs["height"] = downscaled_height * 2
        kwargs.update(**second_pass)

        result = self.video_pipeline(*args, **kwargs)
        if original_output_type != "latent":
            num_frames = result.images.shape[2]
            videos = rearrange(result.images, "b c f h w -> (b f) c h w")

            videos = F.interpolate(
                videos,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            videos = rearrange(videos, "(b f) c h w -> b c f h w", f=num_frames)
            result.images = videos

        return result
