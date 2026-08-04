from .module import Module
from ..parameter import Parameter
from ...tensor import Tensor
import numpy as np

class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
        bias=True
    ):
        super().__init__()
        
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,)
        self.eps = eps
        
        self.weight = None
        self.bias = None
        if elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape))
            if bias:
                self.bias = Parameter(np.zeros(self.normalized_shape))
        
    def forward(self, x: Tensor):
        num_normalized_dim = len(self.normalized_shape)
        
        if x.ndim < num_normalized_dim:
            raise ValueError(
                f"expected input dim to be at least {num_normalized_dim + 1}, but got {x.ndim}"
            )
            
        if x.shape[-num_normalized_dim:] != self.normalized_shape:
            raise ValueError(
                f"expected the last {num_normalized_dim} dim of the input to be {self.normalized_shape}, "
                f"but got {x.shape[-num_normalized_dim:]}"
            )
        
        reduce_dims = tuple(
            -axis for axis in range(1, num_normalized_dim + 1)
        )
        
        mean = x.mean(axis=reduce_dims, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=reduce_dims, keepdims=True)
        
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)
        out = x_norm
        
        if self.weight is not None:
            out *= self.weight
        if self.bias is not None:
            out += self.bias
            
        return out
        
        