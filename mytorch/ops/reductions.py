from ..autograd.function import Function
from ..tensor import Tensor
import numpy as np
from ..autograd.context import Context

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, axis, keepdims):
        ctx.saved_data["x_shape"] = x.data.shape
        ctx.saved_data["axis"] = axis
        ctx.saved_data["keepdims"] = keepdims

        return x.data.sum(axis=axis, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x_shape = ctx.saved_data["x_shape"]
        axis = ctx.saved_data["axis"]
        keepdims = ctx.saved_data["keepdims"]
        
        if axis is not None and not keepdims:
            grad_output = np.expand_dims(
                grad_output,
                axis
            )

        grad_x = np.broadcast_to(
            grad_output,
            x_shape
        )
        
        return grad_x, None, None