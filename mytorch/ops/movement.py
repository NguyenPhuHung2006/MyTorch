from ..autograd.function import Function
from ..autograd.context import Context
import numpy as np

class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, dim0: int, dim1: int):
        ctx.saved_data["dim0"] = dim0
        ctx.saved_data["dim1"] = dim1
        return np.swapaxes(x, dim0, dim1)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        dim0 = ctx.saved_data["dim0"]
        dim1 = ctx.saved_data["dim1"]
        return np.swapaxes(grad_output, dim0, dim1)