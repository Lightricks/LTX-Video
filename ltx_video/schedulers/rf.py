import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
from torch import Tensor

from ltx_video.utils.torch_utils import append_dims


def simple_diffusion_resolution_dependent_timestep_shift(
    samples: Tensor,
    timesteps: Tensor,
    n: int = 32 * 32,
) -> Tensor:
    if len(samples.shape) == 3:
        _, m, _ = samples.shape
    elif len(samples.shape) in [4, 5]:
        m = math.prod(samples.shape[2:])
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
    samples: Tensor, timesteps: Tensor, target_shift_terminal: Optional[float] = None
) -> Tensor:
    """
    Shifts the timestep schedule as a function of the generated resolution.

    In the SD3 paper, the authors empirically how to shift the timesteps based on the resolution of the target images.
    For more details: https://arxiv.org/pdf/2403.03206

    In Flux they later propose a more dynamic resolution dependent timestep shift, see:
    https://github.com/black-forest-labs/flux/blob/87f6fff727a377ea1c378af692afb41ae84cbe04/src/flux/sampling.py#L66


    Args:
        samples (Tensor): A batch of samples with shape (batch_size, channels, height, width) or
            (batch_size, channels, frame, height, width).
        timesteps (Tensor): A batch of timesteps with shape (batch_size,).
        target_shift_terminal (float): The target terminal value for the shifted timesteps.

    Returns:
        Tensor: The shifted timesteps.
    """
    if len(samples.shape) == 3:
        _, m, _ = samples.shape
    elif len(samples.shape) in [4, 5]:
        m = math.prod(samples.shape[2:])
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
    def shift_timesteps(self, samples: Tensor, timesteps: Tensor) -> Tensor:
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
    ):
        super().__init__()
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = self.sigmas = torch.linspace(
            1, 1 / num_train_timesteps, num_train_timesteps
        )
        self.delta_timesteps = self.timesteps - torch.cat(
            [self.timesteps[1:], torch.zeros_like(self.timesteps[-1:])]
        )
        self.shifting = shifting
        self.base_resolution = base_resolution
        self.target_shift_terminal = target_shift_terminal

    def shift_timesteps(self, samples: Tensor, timesteps: Tensor) -> Tensor:
        if self.shifting == "SD3":
            return sd3_resolution_dependent_timestep_shift(
                samples, timesteps, self.target_shift_terminal
            )
        elif self.shifting == "SimpleDiffusion":
            return simple_diffusion_resolution_dependent_timestep_shift(
                samples, timesteps, self.base_resolution
            )
        return timesteps

    def set_timesteps(
        self,
        num_inference_steps: int,
        samples: Tensor,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`): The number of diffusion steps used when generating samples.
            samples (`Tensor`): A batch of samples with shape.
            device (`Union[str, torch.device]`, *optional*): The device to which the timesteps tensor will be moved.
        """
        num_inference_steps = min(self.config.num_train_timesteps, num_inference_steps)
        timesteps = torch.linspace(1, 1 / num_inference_steps, num_inference_steps).to(
            device
        )
        self.timesteps = self.shift_timesteps(samples, timesteps)
        self.delta_timesteps = self.timesteps - torch.cat(
            [self.timesteps[1:], torch.zeros_like(self.timesteps[-1:])]
        )
        self.num_inference_steps = num_inference_steps
        self.sigmas = self.timesteps

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
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[RectifiedFlowSchedulerOutput, Tuple]:
        # pylint: disable=unused-argument
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.FloatTensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.RectifiedFlowSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.rf_scheduler.RectifiedFlowSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if timestep.ndim == 0:
            # Global timestep
            current_index = (self.timesteps - timestep).abs().argmin()
            dt = self.delta_timesteps.gather(0, current_index.unsqueeze(0))
        else:
            # Timestep per token
            assert timestep.ndim == 2
            current_index = (
                (self.timesteps[:, None, None] - timestep[None]).abs().argmin(dim=0)
            )
            dt = self.delta_timesteps[current_index]
            # Special treatment for zero timestep tokens - set dt to 0 so prev_sample = sample
            dt = torch.where(timestep == 0.0, torch.zeros_like(dt), dt)[..., None]

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
