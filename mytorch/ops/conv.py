from ..autograd.function import Function
from ..autograd.context import Context
import numpy as np

class Conv2d(Function):
    @staticmethod
    def forward(ctx: Context, x, weight, bias, stride, padding):
        ...

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray):
        ...