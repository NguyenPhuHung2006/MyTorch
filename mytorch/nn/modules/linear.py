from .module import Module
from ..parameter import Parameter
from ...tensor import Tensor
import numpy as np
from .. import init
import math

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(np.empty((out_features, in_features)))

        if bias:
            self.bias = Parameter(np.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()
            
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            init.uniform_bias_(self.bias, self.weight)
    
    def forward(self, x: Tensor):
        out = x @ self.weight.T
        if self.bias is not None:
            out += self.bias
        return out
        