from ..autograd.function import Function
from ..autograd.context import Context
import numpy as np
from ..utils.broadcast import unbroadcast
from ..utils.linalg import transpose_last_two_dims

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        ctx.save_for_backward(x, y)
        ctx.saved_data["x_shape"] = np.shape(x)
        ctx.saved_data["y_shape"] = np.shape(y)
        ctx.saved_data["x_ndim"] = np.ndim(x)
        ctx.saved_data["y_ndim"] = np.ndim(y)
        return x @ y
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x, y = ctx.saved_tensors
        x_ndim = ctx.saved_data["x_ndim"]
        y_ndim = ctx.saved_data["y_ndim"]
        
        if x_ndim == 1:
            x = np.expand_dims(x, -2)
        if y_ndim == 1:
            y = np.expand_dims(y, -1)
            
        if grad_output.ndim == 0:
            grad_output = grad_output.reshape(1, 1)
        elif x_ndim == 1:
            grad_output = np.expand_dims(grad_output, -2)
        elif y_ndim == 1:
            grad_output = np.expand_dims(grad_output, -1)
        
        grad_x = grad_output @ transpose_last_two_dims(y)
        grad_y = transpose_last_two_dims(x) @ grad_output
        
        if x_ndim == 1:
            grad_x = np.squeeze(grad_x, axis=-2)
        if y_ndim == 1:
            grad_y = np.squeeze(grad_y, axis=-1)
            
        grad_x = unbroadcast(grad_x, ctx.saved_data["x_shape"])
        grad_y = unbroadcast(grad_y, ctx.saved_data["y_shape"])
        
        return grad_x, grad_y
        
        