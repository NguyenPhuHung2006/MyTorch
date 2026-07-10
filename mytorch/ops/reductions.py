from ..autograd.function import Function
import numpy as np
from ..autograd.context import Context
import math

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, axis, keepdims):
        ctx.saved_data["x_shape"] = x.shape
        ctx.saved_data["axis"] = axis
        ctx.saved_data["keepdims"] = keepdims
        return x.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x_shape = ctx.saved_data["x_shape"]
        axis = ctx.saved_data["axis"]
        keepdims = ctx.saved_data["keepdims"]

        if axis is not None and not keepdims:
            grad_output = np.expand_dims(grad_output, axis)

        grad_x = np.broadcast_to(grad_output, x_shape)

        return grad_x, None, None
    
class Mean(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, axis, keepdims):
        ctx.saved_data["x_shape"] = x.shape
        ctx.saved_data["axis"] = axis
        ctx.saved_data["keepdims"] = keepdims
        return x.mean(axis=axis, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x_shape = ctx.saved_data["x_shape"]
        axis = ctx.saved_data["axis"]
        keepdims = ctx.saved_data["keepdims"]
        
        if axis is None:
            div = math.prod(x_shape)
        elif isinstance(axis, tuple):
            div = math.prod(x_shape[a] for a in axis)
        else:
            div = x_shape[axis]

        grad_output = grad_output / div

        if axis is not None and not keepdims:
            grad_output = np.expand_dims(grad_output, axis)

        grad_x = np.broadcast_to(grad_output, x_shape)

        return grad_x, None, None
    
class Max(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, axis, keepdims):
        x_max = x.max(axis=axis, keepdims=keepdims)

        ctx.save_for_backward(x, x_max)
        ctx.saved_data["x_shape"] = x.shape
        ctx.saved_data["axis"] = axis
        ctx.saved_data["keepdims"] = keepdims

        return x_max

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x, x_max = ctx.saved_tensors
        x_shape = ctx.saved_data["x_shape"]
        axis = ctx.saved_data["axis"]
        keepdims = ctx.saved_data["keepdims"]

        if axis is not None and not keepdims:
            grad_output = np.expand_dims(grad_output, axis)
            x_max = np.expand_dims(x_max, axis)

        mask = (x == x_max)

        if axis is None:
            count = mask.sum()
        else:
            count = mask.sum(axis=axis, keepdims=True)

        grad_x = (
            np.broadcast_to(grad_output, x_shape)
            * mask
            / count
        )

        return grad_x, None, None
        
    
class Min(Function):
    @staticmethod
    def forward(ctx: Context, x: np.ndarray, axis, keepdims):
        x_min = x.min(axis=axis, keepdims=keepdims)

        ctx.save_for_backward(x, x_min)
        ctx.saved_data["x_shape"] = x.shape
        ctx.saved_data["axis"] = axis
        ctx.saved_data["keepdims"] = keepdims

        return x_min
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x, x_min = ctx.saved_tensors
        x_shape = ctx.saved_data["x_shape"]
        axis = ctx.saved_data["axis"]
        keepdims = ctx.saved_data["keepdims"]

        if axis is not None and not keepdims:
            grad_output = np.expand_dims(grad_output, axis)
            x_min = np.expand_dims(x_min, axis)

        mask = (x == x_min)

        if axis is None:
            count = mask.sum()
        else:
            count = mask.sum(axis=axis, keepdims=True)

        grad_x = (
            np.broadcast_to(grad_output, x_shape)
            * mask
            / count
        )

        return grad_x, None, None
            
            