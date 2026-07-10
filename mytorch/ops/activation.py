from ..autograd.function import Function
from ..autograd.context import Context
from ..tensor import Tensor
import numpy as np

def relu(x):
    return ReLU.apply(x)

class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, x):
        ctx.saved_data["x_is_tensor"] = isinstance(x, Tensor)
        ctx.saved_data["x_data"] = x.data if isinstance(x, Tensor) else None
        return np.maximum(x, 0)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x_is_tensor = ctx.saved_data["x_is_tensor"]
        x_data = ctx.saved_data["x_data"]
        
        if not x_is_tensor:
            return None
    
        mask = x_data >= 0
        
        return mask
