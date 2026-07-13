from ..autograd.function import Function
from ..autograd.context import Context
from ..tensor import Tensor
import numpy as np
from ..utils.broadcast import unbroadcast

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
    def forward(ctx: Context, x, y):
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
    
class Pow(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        out = x ** y
        ctx.save_for_backward(x, y, out)
        ctx.saved_data["x_shape"] = np.shape(x)
        ctx.saved_data["y_shape"] = np.shape(y)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, out = ctx.saved_tensors

        grad_x = grad_y = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output * y * np.power(x, y - 1)
            grad_x = unbroadcast(grad_x, ctx.saved_data["x_shape"])

        if ctx.needs_input_grad[1]:
            grad_y = grad_output * out * np.log(x)
            grad_y = unbroadcast(grad_y, ctx.saved_data["y_shape"])

        return grad_x, grad_y
    
class TrueDiv(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        out = x / y
        ctx.save_for_backward(y, out)
        ctx.saved_data["x_shape"] = np.shape(x)
        ctx.saved_data["y_shape"] = np.shape(y)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        y, out = ctx.saved_tensors
        grad_x = grad_output / y
        grad_y = -grad_output * out / y
        
        grad_x = unbroadcast(grad_x, ctx.saved_data["x_shape"]) 
        grad_y = unbroadcast(grad_y, ctx.saved_data["y_shape"])
        
        return grad_x, grad_y