from .module import Module
from ..parameter import Parameter
from ...tensor import Tensor
from .. import init
from ...ops.conv import ConvNd as ConvNdFunction

import numpy as np
import math


def _ntuple(n):
    def parse(value):
        if isinstance(value, int):
            return (value,) * n

        if not isinstance(value, tuple):
            raise TypeError(
                f"Expected an int or tuple of length {n}, "
                f"got {type(value).__name__}."
            )

        if len(value) != n:
            raise ValueError(
                f"Expected a tuple of length {n}, "
                f"got {len(value)}."
            )

        return value

    return parse


_pair = _ntuple(2)
_triple = _ntuple(3)


class ConvNd(Module):

    spatial_dims = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        bias: bool = True,
    ):
        super().__init__()

        if self.spatial_dims is None:
            raise ValueError(
                "ConvNd must define spatial_dims."
            )

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be positive."
            )

        if out_channels <= 0:
            raise ValueError(
                "out_channels must be positive."
            )

        # --------------------------------------------------
        # Normalize spatial parameters
        # --------------------------------------------------

        parse = _ntuple(self.spatial_dims)

        self.kernel_size = parse(kernel_size)
        self.stride = parse(stride)
        self.padding = parse(padding)
        self.dilation = parse(dilation)

        # --------------------------------------------------
        # Validate parameters
        # --------------------------------------------------

        for name, values in (
            ("kernel_size", self.kernel_size),
            ("stride", self.stride),
            ("padding", self.padding),
            ("dilation", self.dilation),
        ):
            if any(value < 0 for value in values):
                raise ValueError(
                    f"{name} must contain non-negative values."
                )

        if any(value == 0 for value in self.stride):
            raise ValueError(
                "stride must contain positive values."
            )

        if any(value == 0 for value in self.dilation):
            raise ValueError(
                "dilation must contain positive values."
            )

        # --------------------------------------------------
        # Store channels
        # --------------------------------------------------

        self.in_channels = in_channels
        self.out_channels = out_channels

        # --------------------------------------------------
        # Weight
        #
        # Conv1d:
        # (Cout, Cin, K)
        #
        # Conv2d:
        # (Cout, Cin, Kh, Kw)
        #
        # Conv3d:
        # (Cout, Cin, Kd, Kh, Kw)
        # --------------------------------------------------

        self.weight = Parameter(
            np.empty(
                (
                    out_channels,
                    in_channels,
                    *self.kernel_size,
                )
            )
        )

        # --------------------------------------------------
        # Bias
        # --------------------------------------------------

        if bias:
            self.bias = Parameter(
                np.empty(out_channels)
            )
        else:
            self.bias = None

        # --------------------------------------------------
        # Initialize
        # --------------------------------------------------

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
        )

        if self.bias is not None:
            init.uniform_bias_(
                self.bias,
                self.weight,
            )

    def forward(self, x: Tensor):
        return ConvNdFunction.apply(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )


class Conv1d(ConvNd):

    spatial_dims = 1


class Conv2d(ConvNd):

    spatial_dims = 2


class Conv3d(ConvNd):

    spatial_dims = 3