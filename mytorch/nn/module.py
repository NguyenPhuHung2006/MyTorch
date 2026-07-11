from parameter import Parameter

class Module:
    def __init__(self):
        object.__setattr__(self, "training", True)
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __call__(self, *args):
        return self.forward(*args)
    
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            pass
        elif isinstance(value, Module):
            pass
        else:
            pass
        
    def forward(self, *args):
        raise NotImplementedError

    def parameters(self):
        pass

    def children(self):
        pass

    def modules(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass