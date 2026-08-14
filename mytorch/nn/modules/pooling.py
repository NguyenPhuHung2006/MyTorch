from .module import Module
from ...tensor import Tensor
from ...ops.pooling import PoolNd as PoolNdFunction
from ...utils.ntuple import ntuple
import numpy as np
import math

class PoolNd(Module):

    spatial_dims = None
    reduce_fn = None
    pad_value = None

    def __init__(
        self,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] | None = None,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
    ):
        super().__init__()

        if self.spatial_dims is None or self.reduce_fn is None or self.pad_value is None:
            raise ValueError(
                "PoolNd is an abstract base class."
            )

        # --------------------------------------------------
        # Normalize spatial parameters
        # --------------------------------------------------
        parse = ntuple(self.spatial_dims)

        self.kernel_size = parse(kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        else:
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

    def forward(self, x: Tensor):
        return PoolNdFunction.apply(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.reduce_fn,
            self.backward_fn,
            self.pad_value,
        )

class MaxPoolNd(PoolNd):
    reduce_fn = staticmethod(np.amax)
    pad_value = -np.inf
    
    @staticmethod
    def backward_fn(
        windows: np.ndarray,
        out: np.ndarray,
        pool_axes: tuple,
    ):
        pool_windows = np.expand_dims(out, axis=pool_axes)
        mask = (windows == pool_windows)
        count = mask.sum(axis=pool_axes, keepdims=True)
        grad_factor = mask / count
        return grad_factor
    
class AvgPoolNd(PoolNd):
    reduce_fn = staticmethod(np.mean)
    pad_value = 0
    
    @staticmethod
    def backward_fn(
        windows: np.ndarray,
        out: np.ndarray,
        pool_axes: tuple,
    ):
        kernel_size = tuple(
            windows.shape[axis]
            for axis in pool_axes
        )
        size = math.prod(kernel_size)
        grad_factor = np.full_like(
            windows,
            1 / size,
        )
        return grad_factor

class MaxPool1d(MaxPoolNd):
    spatial_dims = 1

class MaxPool2d(MaxPoolNd):
    spatial_dims = 2

class MaxPool3d(MaxPoolNd):
    spatial_dims = 3
    
class AvgPool1d(AvgPoolNd):
    spatial_dims = 1

class AvgPool2d(AvgPoolNd):
    spatial_dims = 2

class AvgPool3d(AvgPoolNd):
    spatial_dims = 3