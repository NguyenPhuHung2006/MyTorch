from .module import Module
from ...tensor import Tensor

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)
        
    def forward(self, x):
        if not isinstance(x, Tensor):
            raise TypeError(
                f"expected Tensor, got {type(x).__name__}"
            )
        
        for layer in self._modules.values():
            x = layer(x)
        return x
    
    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, index):
        return self._modules[str(index)]