import numpy as np
import pytest

from mytorch.tensor import Tensor
from mytorch.nn.modules.sequential import Sequential
from mytorch.nn.modules.linear import Linear
from mytorch.nn.modules.activations import ReLU
from mytorch.nn.modules.activations import Sigmoid

def test_forward_single_module():
    layer = Linear(3, 2)

    x = Tensor(np.random.randn(5, 3))

    y = Sequential(layer)(x)

    assert y.shape == (5, 2)


def test_forward_multiple_modules():
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2),
    )

    x = Tensor(np.random.randn(8, 3))

    y = model(x)

    assert y.shape == (8, 2)


def test_forward_matches_manual():
    l1 = Linear(3, 4)
    relu = ReLU()
    l2 = Linear(4, 2)

    model = Sequential(l1, relu, l2)

    x = Tensor(np.random.randn(6, 3))

    y1 = model(x)
    y2 = l2(relu(l1(x)))

    np.testing.assert_allclose(y1.data, y2.data)


def test_parameters():
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 1),
    )

    params = list(model.parameters())

    assert len(params) == 4


def test_named_parameters():
    model = Sequential(
        Linear(2, 3),
        Linear(3, 1),
    )

    names = dict(model.named_parameters())

    expected = {
        "0.weight",
        "0.bias",
        "1.weight",
        "1.bias",
    }

    assert set(names.keys()) == expected


def test_train():
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 1),
    )

    model.train()

    assert model.training

    for module in model.modules():
        assert module.training


def test_eval():
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 1),
    )

    model.eval()

    for module in model.modules():
        assert not module.training


def test_zero_grad():
    model = Sequential(
        Linear(2, 3),
        Linear(3, 1),
    )

    for p in model.parameters():
        p.grad = np.random.randn(*p.shape)

    model.zero_grad()

    for p in model.parameters():
        assert p.grad is None


def test_nested_sequential():
    model = Sequential(
        Sequential(
            Linear(2, 3),
            ReLU(),
        ),
        Linear(3, 1),
    )

    x = Tensor(np.random.randn(10, 2))

    y = model(x)

    assert y.shape == (10, 1)


def test_nested_named_parameters():
    model = Sequential(
        Sequential(
            Linear(2, 3),
            Linear(3, 4),
        ),
        Linear(4, 1),
    )

    names = dict(model.named_parameters())

    expected = {
        "0.0.weight",
        "0.0.bias",
        "0.1.weight",
        "0.1.bias",
        "1.weight",
        "1.bias",
    }

    assert set(names.keys()) == expected


def test_empty_sequential():
    model = Sequential()

    x = Tensor(np.random.randn(5, 4))

    y = model(x)

    np.testing.assert_array_equal(y.data, x.data)


@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 8, 32]
)
def test_batch_sizes(batch_size):
    model = Sequential(
        Linear(5, 7),
        ReLU(),
        Linear(7, 4),
    )

    x = Tensor(np.random.randn(batch_size, 5))

    y = model(x)

    assert y.shape == (batch_size, 4)


def test_indexing():
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Sigmoid(),
    )

    assert isinstance(model[0], Linear)
    assert isinstance(model[1], ReLU)
    assert isinstance(model[2], Sigmoid)


def test_iteration():
    l1 = Linear(2, 3)
    relu = ReLU()
    l2 = Linear(3, 1)

    model = Sequential(
        l1,
        relu,
        l2,
    )

    expected = [l1, relu, l2]

    for m1, m2 in zip(model, expected):
        assert m1 is m2