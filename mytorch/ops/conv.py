from ..autograd.function import Function
from ..autograd.context import Context
import numpy as np

class ConvNd(Function):
    @staticmethod
    def forward(ctx: Context, 
                x: np.ndarray, 
                weight: np.ndarray, 
                bias: np.ndarray, 
                stride: tuple, 
                padding: tuple,
                dilation: tuple
            ):
        
        ctx.save_for_backward(x)
        
        batch_size = x.shape[0]
        out_channels = weight.shape[0]
        input_sizes = x.shape[2:]
        kernel_sizes = weight.shape[2:]
        
        output_sizes = tuple(
            (input_size + 2 * pad - dil * (kernel_size - 1) - 1) // stride + 1 
            for input_size, pad, kernel_size, dil, stride 
            in zip(input_sizes, padding, kernel_sizes, dilation, stride)
        )
        
        output_shape = (batch_size, out_channels,) + output_sizes
        
        output = np.zeros(output_shape)
        
        kernel_indices = np.ndindex(kernel_sizes)
        in_channels = weight.shape[1]
        
        for output_index in np.ndindex(output_shape):
            batch_index, out_channel_index, output_size_index = output_index
            
            for kernel_index in kernel_indices:
                for in_channel_index in range(in_channels):
                    input_index = tuple(
                        i * stride + k * dil 
                        for i, stride, k, dil 
                        in zip(output_index, stride, kernel_index, dilation)
                    )
                    
        

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        ...