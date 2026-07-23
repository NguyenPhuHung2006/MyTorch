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
        ...          
        

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        ...