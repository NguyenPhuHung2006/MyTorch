import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, grad_fn=None):
        self.data = np.asarray(data)
        self.grad = None
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        
    def backward(self):
        from .autograd.engine import Engine
        Engine().backward(self)
        
    def __add__(self, other):
        from .ops.arithmetic import Add
        return Add.apply(self, other)
    
    def __sub__(self, other):
        from .ops.arithmetic import Sub
        return Sub.apply(self, other)
    
    def __mul__(self, other):
        from .ops.arithmetic import Mul
        return Mul.apply(self, other)
    
    def __matmul__(self, other):
        pass
    
    def __neg__(self, other):
        pass
    
    def __radd__(self, other):
        pass
    
    def __rsub__(self, other):
        pass
    
    def __rmul__(self, other):
        pass
    
    def __rmatmul__(self, other):
        pass
    
    def __iadd__(self, other):
        pass
    
    def __isub__(self, other):
        pass
    
    def __imul__(self, other):
        pass
    
    def __imatmul__(self, other):
        pass
    
        