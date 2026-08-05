from ..parameter import Parameter

class Module:
    def __init__(self):
        object.__setattr__(self, "training", True)
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

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
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement forward()."
        )

    def parameters(self):
        for module in self.modules():
            yield from module._parameters.values()

    def children(self):
        yield from self._modules.values()

    def modules(self):
        yield self

        for child in self.children():
            yield from child.modules()

    def train(self):
        for module in self.modules():
            module.training = True
        return self

    def eval(self):
        for module in self.modules():
            module.training = False
        return self
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = None
            
    def named_parameters(self):
        def dfs(module, prefix=""):
            for name, param in module._parameters.items():
                yield (f"{prefix}.{name}" if prefix else name), param

            for name, child in module._modules.items():
                yield from dfs(child, f"{prefix}.{name}" if prefix else name)

        yield from dfs(self)