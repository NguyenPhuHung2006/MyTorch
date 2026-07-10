from ..autograd.function import Function
from ..autograd.context import Context
import numpy as np

class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        ctx.save_for_backward(x)
        return np.maximum(x, 0)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x,  = ctx.saved_tensors
        mask = x > 0
        return (mask * grad_output,)
    
class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        out = 1 / (1 + np.exp(-x))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        out, = ctx.saved_tensors
        grad_x = grad_output * out * (1 - out)
        return (grad_x,)
    
class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, dim):
        x_max = x.max(axis=dim, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp_x = exp_x.sum(axis=dim, keepdims=True)
        out = exp_x / sum_exp_x
        
        ctx.save_for_backward(out)
        ctx.saved_data["dim"] = dim
        
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        out, = ctx.saved_tensors
        dim = ctx.saved_data["dim"]
        
        dot = (grad_output * out).sum(axis=dim, keepdims=True)
        grad_x = out * (grad_output - dot)
        
        return (grad_x,)
    
class LogSoftmax(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, dim):
        x_max = x.max(axis=dim, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp_x = exp_x.sum(axis=dim, keepdims=True)
        out = x - x_max - np.log(sum_exp_x)
        
        ctx.save_for_backward(out)
        ctx.saved_data["dim"] = dim
        
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        out, = ctx.saved_tensors
        dim = ctx.saved_data["dim"]
        
        grad_x = grad_output - np.exp(out) * grad_output.sum(axis=dim, keepdims=True)
        
        return (grad_x,)
