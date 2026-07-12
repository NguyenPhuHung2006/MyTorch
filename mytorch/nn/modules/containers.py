from .module import Module

class ModuleList(Module):
    def __init__(self, modules=()):
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)
            
    def append(self, module):
        self.add_module(
            str(len(self)), 
            module
        )
            
    def __iter__(self):
        return iter(self._modules.values())
    
    def __len__(self):
        return len(self._modules)
    
    def __getitem__(self, index):
        return self._modules[str(index)]