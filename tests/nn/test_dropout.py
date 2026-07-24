import numpy as np
import pytest

from mytorch import nn
from mytorch.tensor import Tensor


def test_dropout_p0():
    np.random.seed(0)

    x = Tensor(np.random.randn(3, 4))
    dropout = nn.Dropout(p=0)
    dropout.train()

    y = dropout(x)

    np.testing.assert_allclose(y.data, x.data)


def test_dropout_eval():
    np.random.seed(0)

    x = Tensor(np.random.randn(5, 6))
    dropout = nn.Dropout(0.5)
    dropout.eval()

    y = dropout(x)

    np.testing.assert_allclose(y.data, x.data)


def test_dropout_training():
    np.random.seed(0)

    x = Tensor(np.ones((1000,)))

    dropout = nn.Dropout(0.5)
    dropout.train()

    y = dropout(x)

    assert np.any(y.data == 0)


def test_dropout_scaling():
    np.random.seed(0)

    x = Tensor(np.ones((10000,)))

    dropout = nn.Dropout(0.2)
    dropout.train()

    y = dropout(x)

    surviving = y.data[y.data != 0]

    np.testing.assert_allclose(
        surviving,
        np.full_like(surviving, 1 / 0.8)
    )


def test_dropout_expectation():
    np.random.seed(0)

    x = Tensor(np.ones((100000,)))

    dropout = nn.Dropout(0.3)
    dropout.train()

    y = dropout(x)

    assert abs(y.data.mean() - 1.0) < 0.02


def test_dropout_backward():
    np.random.seed(0)

    x = Tensor(np.ones((1000,)), requires_grad=True)

    dropout = nn.Dropout(0.5)
    dropout.train()

    y = dropout(x)
    y.sum().backward()

    grad = x.grad

    assert np.all(
        np.logical_or(
            np.isclose(grad, 0),
            np.isclose(grad, 2)
        )
    )


def test_dropout_backward_eval():
    np.random.seed(0)

    x = Tensor(np.random.randn(20), requires_grad=True)

    dropout = nn.Dropout(0.5)
    dropout.eval()

    y = dropout(x)
    y.sum().backward()

    np.testing.assert_allclose(x.grad, np.ones_like(x.data))


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_dropout_invalid_probability(p):
    with pytest.raises(ValueError):
        nn.Dropout(p)


def test_dropout1d_channel_dropout():
    np.random.seed(0)

    x = Tensor(np.ones((8, 4, 10)))

    dropout = nn.Dropout1d(0.5)
    dropout.train()

    y = dropout(x)

    for n in range(8):
        for c in range(4):
            channel = y.data[n, c]
            assert np.all(channel == channel[0])


def test_dropout2d_channel_dropout():
    np.random.seed(0)

    x = Tensor(np.ones((4, 3, 5, 5)))

    dropout = nn.Dropout2d(0.5)
    dropout.train()

    y = dropout(x)

    for n in range(4):
        for c in range(3):
            channel = y.data[n, c]
            assert np.all(channel == channel[0, 0])


def test_dropout3d_channel_dropout():
    np.random.seed(0)

    x = Tensor(np.ones((2, 3, 4, 5, 6)))

    dropout = nn.Dropout3d(0.5)
    dropout.train()

    y = dropout(x)

    for n in range(2):
        for c in range(3):
            volume = y.data[n, c]
            assert np.all(volume == volume.flat[0])
            
def test_dropout2d_invalid_input():
    with pytest.raises(ValueError):
        nn.Dropout2d()(Tensor(np.ones((10,))))