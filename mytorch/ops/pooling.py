from ..autograd.function import Function
from ..autograd.context import Context
import numpy as np
from ..utils.window import im2win, win2im

class PoolNd(Function):
    @staticmethod
    def forward(
        ctx: Context,
        x: np.ndarray,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
        dilation: tuple,
        reduce_fn,
        backward_fn,
        pad_value,
    ):
        spatial_ndim = len(kernel_size)
        pool_axes = tuple(
            range(spatial_ndim + 2, 2 + spatial_ndim * 2)
        )
        
        windows = im2win(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            pad_value=pad_value
        )
        
        out = reduce_fn(windows, axis=pool_axes)
        
        ctx.saved_data["kernel_size"] = kernel_size
        ctx.saved_data["stride"] = stride
        ctx.saved_data["padding"] = padding
        ctx.saved_data["dilation"] = dilation
        ctx.saved_data["pool_axes"] = pool_axes
        ctx.saved_data["backward_fn"] = backward_fn
        ctx.saved_data["x_shape"] = x.shape
        
        ctx.save_for_backward(windows, out)
        
        return out
        
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        kernel_size = ctx.saved_data["kernel_size"]
        stride = ctx.saved_data["stride"]
        padding = ctx.saved_data["padding"]
        dilation = ctx.saved_data["dilation"]
        pool_axes = ctx.saved_data["pool_axes"]
        backward_fn = ctx.saved_data["backward_fn"]
        x_shape = ctx.saved_data["x_shape"]
        
        windows, out = ctx.saved_tensors
        
        windows_grad_factor = backward_fn(
            windows, 
            out, 
            pool_axes,
        )
        
        grad_output = np.expand_dims(
            grad_output, 
            axis=pool_axes
        )
        
        windows_grad_x = grad_output * windows_grad_factor
        
        grad_x = win2im(
            windows_grad_x,
            x_shape,
            kernel_size,
            stride,
            padding,
            dilation,
        )
                
        return (grad_x,)