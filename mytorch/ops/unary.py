from ..autograd.function import Function
from ..autograd.context import Context
import numpy as np

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        out = np.exp(x)
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        out, = ctx.saved_tensors
        grad_x = grad_output * out
        return (grad_x,)
    
class Log(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray):
        out = np.log(x)
        ctx.save_for_backward(x)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x, = ctx.saved_tensors
        grad_x = grad_output / x
        return (grad_x,)