from .parameter import Parameter

class Module:
    def __init__(self):
        object.__setattr__(self, "training", True)
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __call__(self, *args):
        return self.forward(*args)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.add_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        object.__setattr__(self, name, value)
            
    def add_module(self, name, module):
        if not isinstance(name, str):
            raise TypeError("module name should be a string")
        if name == "":
            raise ValueError("module name cannot be empty")
        if not isinstance(module, Module):
            raise TypeError(f"{module} is not a Module")
        self._modules[name] = module
        
    def add_parameter(self, name, parameter):
        if not isinstance(name, str):
            raise TypeError("module name should be a string")
        if name == "":
            raise ValueError("module name cannot be empty")
        if not isinstance(parameter, Parameter):
            raise TypeError(f"{parameter} is not a Parameter")
        self._parameters[name] = parameter
        
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