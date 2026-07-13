from ..autograd.function import Function
from ..autograd.context import Context
from ..tensor import Tensor
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
        grad_x = np.swapaxes(grad_output, dim0, dim1)
        return (grad_x,)
    
class GetItem(Function):
    
    @staticmethod
    def unwrap_index(index):
        if isinstance(index, Tensor):
            return index.data
        if isinstance(index, tuple):
            return tuple(GetItem.unwrap_index(i) for i in index)
        if isinstance(index, list):
            return [GetItem.unwrap_index(i) for i in index]
        return index
    
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, index):
        index = GetItem.unwrap_index(index)
        ctx.saved_data["index"] = index
        ctx.saved_data["x_shape"] = x.shape
        return x[index]
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        index = ctx.saved_data["index"]
        x_shape = ctx.saved_data["x_shape"]
        
        grad_x = np.zeros(x_shape, dtype=grad_output.dtype)
        np.add.at(grad_x, index, grad_output)
        
        return (grad_x,) 