from .module import Module
from ..parameter import Parameter
import numpy as np
from .. import init

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
        init.kaiming_uniform_(self.weight, nonlinearity="relu")

        if self.bias is not None:
            init.zeros_(self.bias)
    
    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out += self.bias
        return out
        