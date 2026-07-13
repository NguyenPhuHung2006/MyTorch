import numpy as np
from mytorch.tensor import Tensor
from mytorch.nn.modules.module import Module
from mytorch.nn.modules.linear import Linear
from mytorch.parameter import Parameter

class Dummy(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(np.ones((2, 2)))
        self.bias = Parameter(np.zeros(2))

    def forward(self, x):
        return x


class Parent(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 3)
        self.fc2 = Linear(3, 1)

    def forward(self, x):
        return self.fc2(self.fc1(x))


# ---------------------------------------------------------
# Registration
# ---------------------------------------------------------

def test_register_parameter():
    module = Dummy()

    assert "weight" in module._parameters
    assert "bias" in module._parameters

    assert module._parameters["weight"] is module.weight
    assert module._parameters["bias"] is module.bias


def test_register_submodule():
    model = Parent()

    assert "fc1" in model._modules
    assert "fc2" in model._modules

    assert model._modules["fc1"] is model.fc1
    assert model._modules["fc2"] is model.fc2


# ---------------------------------------------------------
# parameters()
# ---------------------------------------------------------

def test_parameters_returns_all_parameters():
    model = Parent()

    params = list(model.parameters())

    assert len(params) == 4

    assert any(p is model.fc1.weight for p in params)
    assert any(p is model.fc1.bias for p in params)
    assert any(p is model.fc2.weight for p in params)
    assert any(p is model.fc2.bias for p in params)


def test_parameters_are_parameter_objects():
    model = Parent()

    for param in model.parameters():
        assert isinstance(param, Parameter)


# ---------------------------------------------------------
# named_parameters()
# ---------------------------------------------------------

def test_named_parameters():
    model = Parent()

    named = dict(model.named_parameters())

    expected = {
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
    }

    assert set(named.keys()) == expected

    assert named["fc1.weight"] is model.fc1.weight
    assert named["fc1.bias"] is model.fc1.bias
    assert named["fc2.weight"] is model.fc2.weight
    assert named["fc2.bias"] is model.fc2.bias


# ---------------------------------------------------------
# children()
# ---------------------------------------------------------

def test_children():
    model = Parent()

    children = list(model.children())

    assert len(children) == 2

    assert children[0] is model.fc1
    assert children[1] is model.fc2


# ---------------------------------------------------------
# modules()
# ---------------------------------------------------------

def test_modules():
    model = Parent()

    modules = list(model.modules())

    assert model in modules
    assert model.fc1 in modules
    assert model.fc2 in modules

    assert len(modules) == 3


# ---------------------------------------------------------
# train / eval
# ---------------------------------------------------------

def test_train_sets_training_true():
    model = Parent()

    model.eval()

    assert model.training is False
    assert model.fc1.training is False
    assert model.fc2.training is False

    model.train()

    assert model.training is True
    assert model.fc1.training is True
    assert model.fc2.training is True


def test_eval_sets_training_false():
    model = Parent()

    model.eval()

    assert model.training is False
    assert model.fc1.training is False
    assert model.fc2.training is False


# ---------------------------------------------------------
# zero_grad
# ---------------------------------------------------------

def test_zero_grad():
    model = Parent()

    for p in model.parameters():
        p.grad = np.random.randn(*p.shape)

    model.zero_grad()

    for p in model.parameters():
        assert p.grad is None


# ---------------------------------------------------------
# __call__
# ---------------------------------------------------------

def test_call_invokes_forward():
    class AddOne(Module):
        def forward(self, x):
            return x + 1

    model = AddOne()

    x = Tensor(5)

    y = model(x)

    assert y.data == 6


# ---------------------------------------------------------
# nested module recursion
# ---------------------------------------------------------

class Nested(Module):
    def __init__(self):
        super().__init__()

        self.block = Parent()
        self.out = Linear(1, 1)

    def forward(self, x):
        return self.out(self.block(x))


def test_nested_parameters():
    model = Nested()

    params = list(model.parameters())

    assert len(params) == 6


def test_nested_named_parameters():
    model = Nested()

    names = dict(model.named_parameters())

    expected = {
        "block.fc1.weight",
        "block.fc1.bias",
        "block.fc2.weight",
        "block.fc2.bias",
        "out.weight",
        "out.bias",
    }

    assert set(names.keys()) == expected


def test_nested_modules():
    model = Nested()

    modules = list(model.modules())

    assert model in modules
    assert model.block in modules
    assert model.block.fc1 in modules
    assert model.block.fc2 in modules
    assert model.out in modules

    assert len(modules) == 5


def test_nested_train_eval():
    model = Nested()

    model.eval()

    for m in model.modules():
        assert m.training is False

    model.train()

    for m in model.modules():
        assert m.training is True