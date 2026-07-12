from .module import Module
from .containers import ModuleList

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x