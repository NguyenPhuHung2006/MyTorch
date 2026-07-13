import numpy as np
from mytorch.tensor import Tensor
from mytorch.nn.modules.linear import Linear
from mytorch.nn.parameter import Parameter
import pytest

def test_weight_shape():
    layer = Linear(3, 5)

    assert layer.weight.shape == (5, 3)


def test_bias_shape():
    layer = Linear(3, 5)

    assert layer.bias.shape == (5,)


def test_no_bias():
    layer = Linear(3, 5, bias=False)

    assert layer.bias is None


def test_parameter_registration():
    layer = Linear(3, 5)

    params = list(layer.parameters())

    assert len(params) == 2
    assert any(p is layer.weight for p in params)
    assert any(p is layer.bias for p in params)
    

def test_parameter_registration_no_bias():
    layer = Linear(3, 5, bias=False)

    params = list(layer.parameters())

    assert len(params) == 1
    assert layer.weight in params


def test_named_parameters():
    layer = Linear(3, 5)

    names = dict(layer.named_parameters())

    assert set(names.keys()) == {"weight", "bias"}

    assert names["weight"] is layer.weight
    assert names["bias"] is layer.bias


def test_forward_shape_single():
    layer = Linear(3, 5)

    x = Tensor(np.random.randn(3))

    y = layer(x)

    assert y.shape == (5,)


def test_forward_shape_batch():
    layer = Linear(3, 5)

    x = Tensor(np.random.randn(8, 3))

    y = layer(x)

    assert y.shape == (8, 5)


@pytest.mark.parametrize("batch_size", [1, 2, 8, 32])
def test_forward_batch_sizes(batch_size):
    layer = Linear(4, 7)

    x = Tensor(np.random.randn(batch_size, 4))

    y = layer(x)

    assert y.shape == (batch_size, 7)


def test_forward_known_weights():
    layer = Linear(2, 2)

    layer.weight.data = np.array([
        [1., 2.],
        [3., 4.]
    ])

    layer.bias.data = np.array([
        5.,
        6.
    ])

    x = Tensor(np.array([
        [10., 20.]
    ]))

    y = layer(x)

    expected = np.array([
        [
            10 * 1 + 20 * 2 + 5,
            10 * 3 + 20 * 4 + 6
        ]
    ])

    np.testing.assert_allclose(y.data, expected)


def test_forward_no_bias():
    layer = Linear(2, 2, bias=False)

    layer.weight.data = np.array([
        [1., 2.],
        [3., 4.]
    ])

    x = Tensor(np.array([
        [10., 20.]
    ]))

    y = layer(x)

    expected = np.array([
        [
            10 * 1 + 20 * 2,
            10 * 3 + 20 * 4
        ]
    ])

    np.testing.assert_allclose(y.data, expected)


def test_weight_is_parameter():
    layer = Linear(3, 5)

    assert isinstance(layer.weight, Parameter)


def test_bias_is_parameter():
    layer = Linear(3, 5)

    assert isinstance(layer.bias, Parameter)


def test_train():
    layer = Linear(3, 5)

    layer.eval()
    assert not layer.training

    layer.train()
    assert layer.training


def test_eval():
    layer = Linear(3, 5)

    layer.eval()

    assert not layer.training


def test_zero_grad():
    layer = Linear(3, 5)

    layer.weight.grad = np.random.randn(*layer.weight.shape)
    layer.bias.grad = np.random.randn(*layer.bias.shape)

    layer.zero_grad()

    assert layer.weight.grad is None
    assert layer.bias.grad is None