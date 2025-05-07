import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import json
import os
from pathlib import Path

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
from torch import Tensor
from safetensors import safe_open


from ltx_video.utils.torch_utils import append_dims

from ltx_video.utils.diffusers_config_mapping import (
    diffusers_and_ours_config_mapping,
    make_hashable_key,
)


def linear_quadratic_schedule(num_steps, threshold_noise=0.025, linear_steps=None):
    if num_steps == 1:
        return torch.tensor([1.0])
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [
        i * threshold_noise / linear_steps for i in range(linear_steps)
    ]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
        quadratic_steps**2
    )
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule[:-1])


def simple_diffusion_resolution_dependent_timestep_shift(
    samples_shape: torch.Size,
    timesteps: Tensor,
    n: int = 32 * 32,
) -> Tensor:
    if len(samples_shape) == 3:
        _, m, _ = samples_shape
    elif len(samples_shape) in [4, 5]:
        m = math.prod(samples_shape[2:])
    else:
        raise ValueError(
            "Samples must have shape (b, t, c), (b, c, h, w) or (b, c, f, h, w)"
        )
    snr = (timesteps / (1 - timesteps)) ** 2
    shift_snr = torch.log(snr) + 2 * math.log(m / n)
    shifted_timesteps = torch.sigmoid(0.5 * shift_snr)

    return shifted_timesteps


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_normal_shift(
    n_tokens: int,
    min_tokens: int = 1024,
    max_tokens: int = 4096,
    min_shift: float = 0.95,
    max_shift: float = 2.05,
) -> Callable[[float], float]:
    m = (max_shift - min_shift) / (max_tokens - min_tokens)
    b = min_shift - m * min_tokens
    return m * n_tokens + b


def strech_shifts_to_terminal(shifts: Tensor, terminal=0.1):
    """
    Stretch a function (given as sampled shifts) so that its final value matches the given terminal value
    using the provided formula.

    Parameters:
    - shifts (Tensor): The samples of the function to be stretched (PyTorch Tensor).
    - terminal (float): The desired terminal value (value at the last sample).

    Returns:
    - Tensor: The stretched shifts such that the final value equals `terminal`.
    """
    if shifts.numel() == 0:
        raise ValueError("The 'shifts' tensor must not be empty.")

    # Ensure terminal value is valid
    if terminal <= 0 or terminal >= 1:
        raise ValueError("The terminal value must be between 0 and 1 (exclusive).")

    # Transform the shifts using the given formula
    one_minus_z = 1 - shifts
    scale_factor = one_minus_z[-1] / (1 - terminal)
    stretched_shifts = 1 - (one_minus_z / scale_factor)

    return stretched_shifts


def sd3_resolution_dependent_timestep_shift(
    samples_shape: torch.Size,
    timesteps: Tensor,
    target_shift_terminal: Optional[float] = None,
) -> Tensor:
    """
    Shifts the timestep schedule as a function of the generated resolution.

    In the SD3 paper, the authors empirically how to shift the timesteps based on the resolution of the target images.
    For more details: https://arxiv.org/pdf/2403.03206

    In Flux they later propose a more dynamic resolution dependent timestep shift, see:
    https://github.com/black-forest-labs/flux/blob/87f6fff727a377ea1c378af692afb41ae84cbe04/src/flux/sampling.py#L66


    Args:
        samples_shape (torch.Size): The samples batch shape (batch_size, channels, height, width) or
            (batch_size, channels, frame, height, width).
        timesteps (Tensor): A batch of timesteps with shape (batch_size,).
        target_shift_terminal (float): The target terminal value for the shifted timesteps.

    Returns:
        Tensor: The shifted timesteps.
    """
    if len(samples_shape) == 3:
        _, m, _ = samples_shape
    elif len(samples_shape) in [4, 5]:
        m = math.prod(samples_shape[2:])
    else:
        raise ValueError(
            "Samples must have shape (b, t, c), (b, c, h, w) or (b, c, f, h, w)"
        )

    shift = get_normal_shift(m)
    time_shifts = time_shift(shift, 1, timesteps)
    if target_shift_terminal is not None:  # Stretch the shifts to the target terminal
        time_shifts = strech_shifts_to_terminal(time_shifts, target_shift_terminal)
    return time_shifts


class TimestepShifter(ABC):
    @abstractmethod
    def shift_timesteps(self, samples_shape: torch.Size, timesteps: Tensor) -> Tensor:
        pass


@dataclass
class RectifiedFlowSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class RectifiedFlowScheduler(SchedulerMixin, ConfigMixin, TimestepShifter):
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps=1000,
        shifting: Optional[str] = None,
        base_resolution: int = 32**2,
        target_shift_terminal: Optional[float] = None,
        sampler: Optional[str] = "Uniform",
        shift: Optional[float] = None,
    ):
        super().__init__()
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.sampler = sampler
        self.shifting = shifting
        self.base_resolution = base_resolution
        self.target_shift_terminal = target_shift_terminal
        self.timesteps = self.sigmas = self.get_initial_timesteps(
            num_train_timesteps, shift=shift
        )
        self.shift = shift

    def get_initial_timesteps(
        self, num_timesteps: int, shift: Optional[float] = None
    ) -> Tensor:
        if self.sampler == "Uniform":
            return torch.linspace(1, 1 / num_timesteps, num_timesteps)
        elif self.sampler == "LinearQuadratic":
            return linear_quadratic_schedule(num_timesteps)
        elif self.sampler == "Constant":
            assert (
                shift is not None
            ), "Shift must be provided for constant time shift sampler."
            return time_shift(
                shift, 1, torch.linspace(1, 1 / num_timesteps, num_timesteps)
            )

    def shift_timesteps(self, samples_shape: torch.Size, timesteps: Tensor) -> Tensor:
        if self.shifting == "SD3":
            return sd3_resolution_dependent_timestep_shift(
                samples_shape, timesteps, self.target_shift_terminal
            )
        elif self.shifting == "SimpleDiffusion":
            return simple_diffusion_resolution_dependent_timestep_shift(
                samples_shape, timesteps, self.base_resolution
            )
        return timesteps

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        samples_shape: Optional[torch.Size] = None,
        timesteps: Optional[Tensor] = None,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
        If `timesteps` are provided, they will be used instead of the scheduled timesteps.

        Args:
            num_inference_steps (`int` *optional*): The number of diffusion steps used when generating samples.
            samples_shape (`torch.Size` *optional*): The samples batch shape, used for shifting.
            timesteps ('torch.Tensor' *optional*): Specific timesteps to use instead of scheduled timesteps.
            device (`Union[str, torch.device]`, *optional*): The device to which the timesteps tensor will be moved.
        """
        if timesteps is not None and num_inference_steps is not None:
            raise ValueError(
                "You cannot provide both `timesteps` and `num_inference_steps`."
            )
        if timesteps is None:
            num_inference_steps = min(
                self.config.num_train_timesteps, num_inference_steps
            )
            timesteps = self.get_initial_timesteps(
                num_inference_steps, shift=self.shift
            ).to(device)
            timesteps = self.shift_timesteps(samples_shape, timesteps)
        else:
            timesteps = torch.Tensor(timesteps).to(device)
            num_inference_steps = len(timesteps)
        self.timesteps = timesteps
        self.num_inference_steps = num_inference_steps
        self.sigmas = self.timesteps

    @staticmethod
    def from_pretrained(pretrained_model_path: Union[str, os.PathLike]):
        pretrained_model_path = Path(pretrained_model_path)
        if pretrained_model_path.is_file():
            comfy_single_file_state_dict = {}
            with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for k in f.keys():
                    comfy_single_file_state_dict[k] = f.get_tensor(k)
            configs = json.loads(metadata["config"])
            config = configs["scheduler"]
            del comfy_single_file_state_dict

        elif pretrained_model_path.is_dir():
            diffusers_noise_scheduler_config_path = (
                pretrained_model_path / "scheduler" / "scheduler_config.json"
            )

            with open(diffusers_noise_scheduler_config_path, "r") as f:
                scheduler_config = json.load(f)
            hashable_config = make_hashable_key(scheduler_config)
            if hashable_config in diffusers_and_ours_config_mapping:
                config = diffusers_and_ours_config_mapping[hashable_config]
        return RectifiedFlowScheduler.from_config(config)

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Optional[int] = None
    ) -> torch.FloatTensor:
        # pylint: disable=unused-argument
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.FloatTensor,
        sample: torch.FloatTensor,
        return_dict: bool = True,
        stochastic_sampling: Optional[bool] = False,
        **kwargs,
    ) -> Union[RectifiedFlowSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).
        z_{t_1} = z_t - \Delta_t * v
        The method finds the next timestep that is lower than the input timestep(s) and denoises the latents
        to that level. The input timestep(s) are not required to be one of the predefined timesteps.

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model - the velocity,
            timestep (`float`):
                The current discrete timestep in the diffusion chain (global or per-token).
            sample (`torch.FloatTensor`):
                A current latent tokens to be de-noised.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.
            stochastic_sampling (`bool`, *optional*, defaults to `False`):
                Whether to use stochastic sampling for the sampling process.

        Returns:
            [`~schedulers.scheduling_utils.RectifiedFlowSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.rf_scheduler.RectifiedFlowSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        t_eps = 1e-6  # Small epsilon to avoid numerical issues in timestep values

        timesteps_padded = torch.cat(
            [self.timesteps, torch.zeros(1, device=self.timesteps.device)]
        )

        # Find the next lower timestep(s) and compute the dt from the current timestep(s)
        if timestep.ndim == 0:
            # Global timestep case
            lower_mask = timesteps_padded < timestep - t_eps
            lower_timestep = timesteps_padded[lower_mask][0]  # Closest lower timestep
            dt = timestep - lower_timestep

        else:
            # Per-token case
            assert timestep.ndim == 2
            lower_mask = timesteps_padded[:, None, None] < timestep[None] - t_eps
            lower_timestep = lower_mask * timesteps_padded[:, None, None]
            lower_timestep, _ = lower_timestep.max(dim=0)
            dt = (timestep - lower_timestep)[..., None]

        # Compute previous sample
        if stochastic_sampling:
            x0 = sample - timestep[..., None] * model_output
            next_timestep = timestep[..., None] - dt
            prev_sample = self.add_noise(x0, torch.randn_like(sample), next_timestep)
        else:
            prev_sample = sample - dt * model_output

        if not return_dict:
            return (prev_sample,)

        return RectifiedFlowSchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        sigmas = timesteps
        sigmas = append_dims(sigmas, original_samples.ndim)
        alphas = 1 - sigmas
        noisy_samples = alphas * original_samples + sigmas * noise
        return noisy_samples
