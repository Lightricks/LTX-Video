import torch
from torch import nn


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    elif dims_to_append == 0:
        return x
    return x[(...,) + (None,) * dims_to_append]


class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive."""

    def __init__(self, *args, **kwargs) -> None:  # pylint: disable=unused-argument
        super().__init__()

    # pylint: disable=unused-argument
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x
