from .module import Module
from ..tensor import Tensor

class ReLU(Module):
    def forward(self, x: Tensor):
        super().__init__()
        return x.relu()
    
class Sigmoid(Module):
    def forward(self, x: Tensor):
        super().__init__()
        return x.sigmoid()
    
class Softmax(Module):
    def forward(self, x: Tensor):
        super().__init__()
        return x.softmax()
    
class LogSoftmax(Module):
    def forward(self, x: Tensor):
        super().__init__()
        return x.log_softmax()