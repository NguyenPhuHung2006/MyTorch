from .module import Module
from ...parameter import Parameter
import numpy as np

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        bound = 1 / np.sqrt(in_features)
        self.weight = Parameter(
            np.random.uniform(
                -bound,
                bound,
                size=(out_features, in_features)
            )
        )
        
        if bias:
            self.bias = Parameter(
            np.random.uniform(
                -bound,
                bound,
                size=(out_features,)
            )
        )
        else:
            self.bias = None
    
    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out += self.bias
        return out
        