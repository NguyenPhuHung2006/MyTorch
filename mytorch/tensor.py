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
    
    def __neg__(self):
        return self * -1
    
    def __pow__(self, other):
        from .ops.arithmetic import Pow
        return Pow.apply(self, other)
    
    def __truediv__(self, other):
        from .ops.arithmetic import TrueDiv
        return TrueDiv.apply(self, other)
    
    def __radd__(self, other):
        from .ops.arithmetic import Add
        return Add.apply(self, other)
    
    def __rsub__(self, other):
        from .ops.arithmetic import Sub
        return Sub.apply(other, self)
    
    def __rmul__(self, other):
        from .ops.arithmetic import Mul
        return Mul.apply(self, other)
    
    def __rmatmul__(self, other):
        from .ops.linalg import MatMul
        return MatMul.apply(other, self)
    
    def __iadd__(self, other):
        from .ops.arithmetic import Add
        return Add.apply(self, other)
    
    def __isub__(self, other):
        from .ops.arithmetic import Sub
        return Sub.apply(self, other)
    
    def __imul__(self, other):
        from .ops.arithmetic import Mul
        return Mul.apply(self, other)
    
    def __imatmul__(self, other):
        from .ops.linalg import MatMul
        return MatMul.apply(self, other)
    
    def __lt__(self, other):
        from .ops.comparison import Lt
        return Lt.apply(self, other)
    
    def __le__(self, other):
        from .ops.comparison import Le
        return Le.apply(self, other)
    
    def __gt__(self, other):
        from .ops.comparison import Gt
        return Gt.apply(self, other)
    
    def __ge__(self, other):
        from .ops.comparison import Ge
        return Ge.apply(self, other)
    
    def __eq__(self, value):
        from .ops.comparison import Eq
        return Eq.apply(self, value)
    
    def __ne__(self, value):
        from .ops.comparison import Ne
        return Ne.apply(self, value)
    
    def __hash__(self):
        return id(self)
    
    def __bool__(self):
        if self.numel() != 1:
            raise RuntimeError(
                "Boolean value of Tensor with more than one value is ambiguous"
            )

        return bool(self.data.item())
    
    def __getitem__(self, index):
        from .ops.movement import GetItem
        return GetItem.apply(self, index)
    
    def numel(self):
        return np.size(self.data)

    def item(self):
        if self.data.size != 1:
            raise ValueError(
                "Can only convert a Tensor with one element to a Python scalar."
            )
        return self.data.item()
    
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        from .ops.reduction import Sum
        return Sum.apply(self, axis, keepdims)
    
    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        from .ops.reduction import Mean
        return Mean.apply(self, axis, keepdims)
    
    def max(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        from .ops.reduction import Max
        return Max.apply(self, axis, keepdims)
    
    def min(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        from .ops.reduction import Min
        return Min.apply(self, axis, keepdims)
    
    def any(self, dim: int | None = None, keepdims: bool = False):
        from .ops.reduction import Any
        return Any.apply(self, dim, keepdims)
    
    def relu(self):
        from .ops.activation import ReLU
        return ReLU.apply(self)
    
    def sigmoid(self):
        from .ops.activation import Sigmoid
        return Sigmoid.apply(self)
    
    def tanh(self):
        from .ops.activation import Tanh
        return Tanh.apply(self)
    
    def softmax(self, dim: int | None = -1):
        from .ops.activation import Softmax
        return Softmax.apply(self, dim)
    
    def log_softmax(self, dim: int | None = -1):
        from .ops.activation import LogSoftmax
        return LogSoftmax.apply(self, dim)
    
    def exp(self):
        from .ops.unary import Exp
        return Exp.apply(self)
    
    def log(self):
        from .ops.unary import Log
        return Log.apply(self)
    
    @property
    def T(self):
        if (self.data.ndim < 2):
            return self
        from .ops.movement import Transpose
        return Transpose.apply(self, -2, -1)
    
    def transpose(self, dim0: int = -2, dim1: int = -1):
        if (self.data.ndim < 2):
            return self
        from .ops.movement import Transpose
        return Transpose.apply(self, dim0, dim1)
    
    def reshape(self, *shape):
        if len(shape) == 1:
            shape = shape[0]
        from .ops.movement import Reshape
        return Reshape.apply(self, shape)
    
    def flatten(self, start_dim=0, end_dim=-1):
        shape = self.shape
        ndim = len(shape)
        
        if not (-ndim <= start_dim < ndim):
            raise IndexError("start_dim out of range")
        if not (-ndim <= end_dim < ndim):
            raise IndexError("end_dim out of range")
        
        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim
            
        if start_dim > end_dim:
            raise ValueError("start_dim must be <= end_dim")
        
        flattened = 1
        for s in shape[start_dim:end_dim+1]:
            flattened *= s
        
        new_shape = shape[:start_dim] + (flattened,) + shape[end_dim+1:]
        return self.reshape(new_shape)
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype
    
    def numpy(self):
        return self.data
        