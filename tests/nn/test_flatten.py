import numpy as np

from mytorch import Tensor
from mytorch.nn import Flatten


def test_flatten_module_default():
    layer = Flatten()

    x = Tensor(np.zeros((8, 1, 28, 28)))

    y = layer(x)

    assert y.shape == (8, 784)


def test_flatten_module_custom():
    layer = Flatten(start_dim=2)

    x = Tensor(np.zeros((2, 3, 4, 5)))

    y = layer(x)

    assert y.shape == (2, 3, 20)


def test_flatten_module_backward():
    layer = Flatten()

    x = Tensor(np.random.randn(4, 3, 5), requires_grad=True)

    y = layer(x)

    loss = y.sum()

    loss.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.ones((4, 3, 5))
    )