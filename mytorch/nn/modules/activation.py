from .module import Module
from ...tensor import Tensor

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