from ..autograd.function import Function
from ..autograd.context import Context
from ..tensor import Tensor
from numpy import np

class Add(Function): 
    
    @staticmethod
    def forward(ctx: Context, x: Tensor, y: Tensor):
        ctx.save_for_backward(x, y)
        return x.data + y.data
        
    @staticmethod
    def backward(ctx: Context, grad_output):
        pass
        

class Sub(Function):
    ...

class Mul(Function):
    ...