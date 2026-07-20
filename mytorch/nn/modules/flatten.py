from .module import Module
from ...tensor import Tensor 

class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor):
        return x.flatten(self.start_dim, self.end_dim)