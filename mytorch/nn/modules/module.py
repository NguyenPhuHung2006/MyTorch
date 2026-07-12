from ...parameter import Parameter

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
        
    def _validate_name(self, name):
        if not isinstance(name, str):
            raise TypeError("name should be a string")

        if name == "":
            raise ValueError("name cannot be empty")

        if "." in name:
            raise KeyError("name cannot contain '.'")
            
    def add_module(self, name, module):
        self._validate_name(name)
        if not isinstance(module, Module):
            raise TypeError(f"{module} is not a Module")
        self._modules[name] = module
        
    def add_parameter(self, name, parameter):
        self._validate_name(name)
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