from ..autograd.function import Function
from ..autograd.context import Context
from ..tensor import Tensor
import numpy as np
from ..autograd.utils import unbroadcast

class Add(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y):
        ctx.saved_data["y_is_tensor"] = isinstance(y, Tensor)
        ctx.saved_data["x_shape"] = x.data.shape
        ctx.saved_data["y_shape"] = (
            y.data.shape if isinstance(y, Tensor) else None
        )
        return x.data + (y.data if isinstance(y, Tensor) else y)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        grad_x = unbroadcast(
            grad_output,
            ctx.saved_data["x_shape"]
        )

        grad_y = None
        if ctx.saved_data["y_is_tensor"]:
            grad_y = unbroadcast(
                grad_output,
                ctx.saved_data["y_shape"]
            )

        return grad_x, grad_y

class Sub(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y):
        ctx.saved_data["y_is_tensor"] = isinstance(y, Tensor)
        ctx.saved_data["x_shape"] = x.data.shape
        ctx.saved_data["y_shape"] = (
            y.data.shape if isinstance(y, Tensor) else None
        )
        return x.data - (y.data if isinstance(y, Tensor) else y)
        
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        grad_x = unbroadcast(
            grad_output,
            ctx.saved_data["x_shape"]
        )

        grad_y = None
        if ctx.saved_data["y_is_tensor"]:
            grad_y = unbroadcast(
                -grad_output,
                ctx.saved_data["y_shape"]
            )

        return grad_x, grad_y

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y):
        ctx.saved_data["x_data"] = x.data
        ctx.saved_data["y"] = y
        ctx.saved_data["y_is_tensor"] = isinstance(y, Tensor)
        ctx.saved_data["x_shape"] = x.data.shape
        ctx.saved_data["y_shape"] = (
            y.data.shape if isinstance(y, Tensor) else None
        )
        return x.data * (y.data if isinstance(y, Tensor) else y)
        
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x_data = ctx.saved_data["x_data"]
        y = ctx.saved_data["y"]
        y_is_tensor = ctx.saved_data["y_is_tensor"]
        
        grad_x = grad_output * (
            y.data if y_is_tensor else y
        )
        grad_y = None
        if y_is_tensor:
            grad_y = grad_output * x_data
            
        grad_x = unbroadcast(
            grad_x,
            ctx.saved_data["x_shape"]
        )

        if y_is_tensor and grad_y is not None:
            grad_y = unbroadcast(
                grad_y,
                ctx.saved_data["y_shape"]
            )

        return grad_x, grad_y