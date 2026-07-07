import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, grad_fn=None):
        self.data = np.asarray(data)
        self.grad = None
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        
    def backward(self, init_grad: np.ndarray | None = None):
        from .autograd.engine import Engine
        Engine().backward(self, init_grad)
        
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
        from .ops.linalg import MatMul
        return MatMul.apply(self, other)
    
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
    
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        from .ops.reductions import Sum
        return Sum.apply(self, axis, keepdims)
    
    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        from .ops.reductions import Mean
        return Mean.apply(self, axis, keepdims)
    
    def max(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        from .ops.reductions import Max
        return Max.apply(self, axis, keepdims)
    
    def min(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        from .ops.reductions import Min
        return Min.apply(self, axis, keepdims)
        