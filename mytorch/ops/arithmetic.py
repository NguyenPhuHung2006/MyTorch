from ..autograd.function import Function
from ..autograd.context import Context
from ..tensor import Tensor
import numpy as np
from ..autograd.utils import unbroadcast

class Add(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        ctx.saved_data["x_shape"] = np.shape(x)
        ctx.saved_data["y_shape"] = np.shape(y)
        return x + y

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        grad_x = unbroadcast(grad_output, ctx.saved_data["x_shape"])
        grad_y = unbroadcast(grad_output, ctx.saved_data["y_shape"])
        return grad_x, grad_y

class Sub(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y):
        ctx.saved_data["x_shape"] = np.shape(x)
        ctx.saved_data["y_shape"] = np.shape(y)
        return x - y
        
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        grad_x = unbroadcast(grad_output, ctx.saved_data["x_shape"])
        grad_y = unbroadcast(-grad_output, ctx.saved_data["y_shape"])
        return grad_x, grad_y

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        ctx.save_for_backward(x, y)
        ctx.saved_data["x_shape"] = np.shape(x)
        ctx.saved_data["y_shape"] = np.shape(y)
        return x * y

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x, y = ctx.saved_tensors

        grad_x = grad_output * y
        grad_y = grad_output * x

        grad_x = unbroadcast(grad_x, ctx.saved_data["x_shape"])
        grad_y = unbroadcast(grad_y, ctx.saved_data["y_shape"])

        return grad_x, grad_y