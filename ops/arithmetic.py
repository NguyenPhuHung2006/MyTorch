from ..autograd.function import Function
from ..autograd.context import Context
from ..tensor import Tensor
import numpy as np

class Add(Function):
    @staticmethod
    def forward(ctx, x: Tensor, y):
        ctx.saved_data["y_is_tensor"] = isinstance(y, Tensor)
        ctx.saved_data["x_shape"] = x.data.shape
        ctx.saved_data["y_shape"] = (
            y.data.shape if isinstance(y, Tensor) else None
        )
        return x.data + (y.data if isinstance(y, Tensor) else y)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        grad_x = grad_output
        grad_y = (
            grad_output
            if ctx.saved_data["y_is_tensor"]
            else None
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
        grad_x = grad_output
        grad_y = (
            -grad_output
            if ctx.saved_data["y_is_tensor"]
            else None
        )
        
        return grad_x, grad_y

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y):
        ctx.save_for_backward(x)
        ctx.saved_data["y"] = y
        ctx.saved_data["x_shape"] = x.data.shape
        ctx.saved_data["y_shape"] = (
            y.data.shape if isinstance(y, Tensor) else None
        )
        return x.data * (y.data if isinstance(y, Tensor) else y)
        
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x, = ctx.saved_tensors
        y = ctx.saved_data["y"]
        grad_x = grad_output * (
            y.data if isinstance(y, Tensor) else y
        )
        grad_y = None
        if isinstance(y, Tensor):
            grad_y = grad_output * x.data
            
        return grad_x, grad_y