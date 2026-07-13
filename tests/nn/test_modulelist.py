import numpy as np
from mytorch.nn.modules.containers import ModuleList
from mytorch.nn.modules.linear import Linear
from mytorch.nn.modules.activations import ReLU

def test_len():
    modules = ModuleList([
        Linear(2, 3),
        ReLU(),
        Linear(3, 1),
    ])

    assert len(modules) == 3


def test_indexing():
    l1 = Linear(2, 3)
    relu = ReLU()
    l2 = Linear(3, 1)

    modules = ModuleList([l1, relu, l2])

    assert modules[0] is l1
    assert modules[1] is relu
    assert modules[2] is l2


def test_iteration():
    l1 = Linear(2, 3)
    relu = ReLU()
    l2 = Linear(3, 1)

    modules = ModuleList([l1, relu, l2])

    expected = [l1, relu, l2]

    for m1, m2 in zip(modules, expected):
        assert m1 is m2


def test_append():
    modules = ModuleList()

    l1 = Linear(2, 3)
    modules.append(l1)

    assert len(modules) == 1
    assert modules[0] is l1


def test_extend():
    modules = ModuleList()

    l1 = Linear(2, 3)
    relu = ReLU()
    l2 = Linear(3, 1)

    modules.extend([l1, relu, l2])

    assert len(modules) == 3
    assert modules[0] is l1
    assert modules[1] is relu
    assert modules[2] is l2

def test_parameters():
    l1 = Linear(2, 3)
    l2 = Linear(3, 1)

    modules = ModuleList([l1, l2])

    params = list(modules.parameters())

    assert len(params) == 4

    assert any(p is l1.weight for p in params)
    assert any(p is l1.bias for p in params)
    assert any(p is l2.weight for p in params)
    assert any(p is l2.bias for p in params)

def test_named_parameters():
    modules = ModuleList([
        Linear(2, 3),
        Linear(3, 1),
    ])

    names = dict(modules.named_parameters())

    expected = {
        "0.weight",
        "0.bias",
        "1.weight",
        "1.bias",
    }

    assert set(names.keys()) == expected


def test_nested_modulelist_parameters():
    inner = ModuleList([
        Linear(2, 3),
        Linear(3, 1),
    ])

    outer = ModuleList([
        inner,
        Linear(1, 1),
    ])

    params = list(outer.parameters())

    assert len(params) == 6


def test_nested_modulelist_named_parameters():
    inner = ModuleList([
        Linear(2, 3),
        Linear(3, 1),
    ])

    outer = ModuleList([
        inner,
        Linear(1, 1),
    ])

    names = dict(outer.named_parameters())

    expected = {
        "0.0.weight",
        "0.0.bias",
        "0.1.weight",
        "0.1.bias",
        "1.weight",
        "1.bias",
    }

    assert set(names.keys()) == expected


def test_train():
    modules = ModuleList([
        Linear(2, 3),
        ReLU(),
        Linear(3, 1),
    ])

    modules.train()

    assert modules.training

    for m in modules:
        assert m.training


def test_eval():
    modules = ModuleList([
        Linear(2, 3),
        ReLU(),
        Linear(3, 1),
    ])

    modules.eval()

    assert not modules.training

    for m in modules:
        assert not m.training


def test_zero_grad():
    modules = ModuleList([
        Linear(2, 3),
        Linear(3, 1),
    ])

    for p in modules.parameters():
        p.grad = np.random.randn(*p.shape)

    modules.zero_grad()

    for p in modules.parameters():
        assert p.grad is None


def test_empty_modulelist():
    modules = ModuleList()

    assert len(modules) == 0
    assert list(modules) == []
    assert list(modules.parameters()) == []