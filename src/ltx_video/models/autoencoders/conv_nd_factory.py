from typing import Tuple, Union

import torch

from ltx_video.models.autoencoders.dual_conv3d import DualConv3d
from ltx_video.models.autoencoders.causal_conv3d import CausalConv3d


def make_conv_nd(
    dims: Union[int, Tuple[int, int]],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    causal=False,
    spatial_padding_mode="zeros",
    temporal_padding_mode="zeros",
):
    if not (spatial_padding_mode == temporal_padding_mode or causal):
        raise NotImplementedError("spatial and temporal padding modes must be equal")
    if dims == 2:
        return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode,
        )
    elif dims == 3:
        if causal:
            return CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                spatial_padding_mode=spatial_padding_mode,
            )
        return torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode,
        )
    elif dims == (2, 1):
        return DualConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def make_linear_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    bias=True,
):
    if dims == 2:
        return torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias
        )
    elif dims == 3 or dims == (2, 1):
        return torch.nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias
        )
    else:
        raise ValueError(f"unsupported dimensions: {dims}")
