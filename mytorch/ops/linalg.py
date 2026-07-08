from ..autograd.function import Function
from ..autograd.context import Context
from ..tensor import Tensor
import numpy as np
from ..autograd.utils import matrix_transpose, unbroadcast

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, y):
        ctx.saved_data["x_data"] = x.data
        ctx.saved_data["y_data"] = y.data if isinstance(y, Tensor) else y;
        ctx.saved_data["y_is_tensor"] = isinstance(y, Tensor)
        ctx.saved_data["x_shape"] = x.data.shape
        ctx.saved_data["y_shape"] = (
            y.data.shape if isinstance(y, Tensor) else None
        )
        ctx.saved_data["x_ndim"] = x.data.ndim
        ctx.saved_data["y_ndim"] = y.data.ndim if isinstance(y, Tensor) else np.asarray(y).ndim
        return x.data @ (y.data if isinstance(y, Tensor) else y)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        x_data = ctx.saved_data["x_data"]
        y_data = ctx.saved_data["y_data"]
        y_is_tensor = ctx.saved_data["y_is_tensor"]
        x_ndim = ctx.saved_data["x_ndim"]
        y_ndim = ctx.saved_data["y_ndim"]
        
        if x_ndim == 1:
            x_data = np.expand_dims(x_data, -2)
        if y_ndim == 1:
            y_data = np.expand_dims(y_data, -1)
            
        if grad_output.ndim == 0:
            grad_output = grad_output.reshape(1, 1)
        elif x_ndim == 1:
            grad_output = np.expand_dims(grad_output, -2)
        elif y_ndim == 1:
            grad_output = np.expand_dims(grad_output, -1)
        
        grad_x = grad_output @ matrix_transpose(y_data)
        grad_y = matrix_transpose(x_data) @ grad_output if y_is_tensor else None
        
        if x_ndim == 1:
            grad_x = np.squeeze(grad_x, axis=-2)
        if y_ndim == 1:
            grad_y = np.squeeze(grad_y, axis=-1)
            
        grad_x = unbroadcast(grad_x, ctx.saved_data["x_shape"])
        if y_is_tensor:
            grad_y = unbroadcast(grad_y, ctx.saved_data["y_shape"])
        
        return grad_x, grad_y
        
        