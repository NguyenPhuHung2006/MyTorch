from .module import Module
from ...tensor import Tensor
import numpy as np

class ReLU(Module):
    def forward(self, x: Tensor):
        return x.relu()
    
class Sigmoid(Module):
    def forward(self, x: Tensor):
        return x.sigmoid()
    
class Tanh(Module):
    def forward(self, x: Tensor):
        return x.tanh()
    
class Softmax(Module):
    def __init__(self, dim: int | None = -1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: Tensor):
        return x.softmax(self.dim)
    
class LogSoftmax(Module):
    def __init__(self, dim: int | None = -1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: Tensor):
        return x.log_softmax(self.dim)
    
class GELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (
            1 + Tanh()(
                np.sqrt(2 / np.pi) *
                (x + 0.044715 * x ** 3)
            )
        )