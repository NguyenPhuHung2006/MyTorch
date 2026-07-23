from .module import Module
from ..parameter import Parameter
from ...tensor import Tensor
import numpy as np
from .. import init
from ...ops.conv import ConvNd as ConvNdFunction
import math

def _pair(value):
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Expected a tuple of length 2.")
        return value
    return (value, value)

class Conv2d(Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: int | tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = True):
        super().__init__()
        
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        kernel_h, kernel_w = self.kernel_size
        
        self.weight = Parameter(np.empty((
            out_channels, 
            in_channels, 
            kernel_h, 
            kernel_w
        )))
        
        if bias:
            self.bias = Parameter(np.empty(out_channels))
        else:
            self.bias = None
            
        self.reset_parameters()
            
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            init.uniform_bias_(self.bias, self.weight)
            
    def forward(self, x: Tensor):
        return ConvNdFunction.apply(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation
        )
            
    