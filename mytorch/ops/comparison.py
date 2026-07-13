from ..autograd.function import Function
from ..autograd.context import Context
import numpy as np

class Comparison(Function):
    differentiable = False
    
    @staticmethod
    def backward(ctx: Context, grad_out: np.ndarray):
        raise RuntimeError("Comparison operations do not support backward.")

class Le(Comparison):
    @staticmethod
    def forward(ctx: Context, x, y):
        return x <= y

class Lt(Comparison):
    @staticmethod
    def forward(ctx: Context, x, y):
        return x < y
    
class Ge(Comparison):
    @staticmethod
    def forward(ctx: Context, x, y):
        return x >= y
    
class Gt(Comparison):
    @staticmethod
    def forward(ctx: Context, x, y):
        return x > y
    
class Eq(Comparison):
    @staticmethod
    def forward(ctx: Context, x, y):
        return x == y
    
class Ne(Comparison):
    @staticmethod
    def forward(ctx: Context, x, y):
        return x != y