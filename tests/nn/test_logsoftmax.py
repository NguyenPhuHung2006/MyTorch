import numpy as np

from mytorch.tensor import Tensor
from mytorch.nn.modules.activations import LogSoftmax

def test_logsoftmax_forward():
    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    logsoftmax = LogSoftmax(dim=1)

    y = logsoftmax(x)

    x_shift = x.data - np.max(x.data, axis=1, keepdims=True)
    expected = x_shift - np.log(np.exp(x_shift).sum(axis=1, keepdims=True))

    np.testing.assert_allclose(y.data, expected, rtol=1e-6)


def test_logsoftmax_shape():
    x = Tensor(np.random.randn(3, 4, 5))
    logsoftmax = LogSoftmax(dim=-1)

    y = logsoftmax(x)

    assert y.shape == x.shape


def test_logsoftmax_exp_sum_to_one():
    x = Tensor(np.random.randn(8, 10))
    logsoftmax = LogSoftmax(dim=1)

    y = logsoftmax(x)

    np.testing.assert_allclose(
        np.exp(y.data).sum(axis=1),
        np.ones(8),
        rtol=1e-6,
    )


def test_logsoftmax_dim0():
    x = Tensor(np.random.randn(5, 3))
    logsoftmax = LogSoftmax(dim=0)

    y = logsoftmax(x)

    np.testing.assert_allclose(
        np.exp(y.data).sum(axis=0),
        np.ones(3),
        rtol=1e-6,
    )


def test_logsoftmax_dim_negative():
    x = Tensor(np.random.randn(2, 3, 4))
    logsoftmax = LogSoftmax(dim=-1)

    y = logsoftmax(x)

    np.testing.assert_allclose(
        np.exp(y.data).sum(axis=-1),
        np.ones((2, 3)),
        rtol=1e-6,
    )


def test_logsoftmax_backward():
    x = Tensor(
        np.random.randn(3, 4),
        requires_grad=True,
    )

    logsoftmax = LogSoftmax(dim=1)

    y = logsoftmax(x)

    upstream = np.random.randn(*y.shape)

    y.backward(upstream)

    softmax = np.exp(y.data)
    expected = upstream - softmax * np.sum(
        upstream,
        axis=1,
        keepdims=True,
    )

    np.testing.assert_allclose(
        x.grad,
        expected,
        rtol=1e-6,
    )


def test_logsoftmax_no_parameters():
    logsoftmax = LogSoftmax()

    assert list(logsoftmax.parameters()) == []


def test_logsoftmax_train_eval():
    logsoftmax = LogSoftmax()

    logsoftmax.eval()
    assert logsoftmax.training is False

    logsoftmax.train()
    assert logsoftmax.training is True
    
def test_logsoftmax_gradient_numerical():
    eps = 1e-6

    x_data = np.random.randn(2, 3)

    x = Tensor(x_data.copy(), requires_grad=True)

    logsoftmax = LogSoftmax(dim=1)
    y = logsoftmax(x)

    upstream = np.random.randn(*y.shape)

    y.backward(upstream)

    numerical = np.zeros_like(x_data)

    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            plus = x_data.copy()
            minus = x_data.copy()

            plus[i, j] += eps
            minus[i, j] -= eps

            plus_shift = plus - plus.max(axis=1, keepdims=True)
            minus_shift = minus - minus.max(axis=1, keepdims=True)

            y1 = plus_shift - np.log(
                np.exp(plus_shift).sum(axis=1, keepdims=True)
            )
            y2 = minus_shift - np.log(
                np.exp(minus_shift).sum(axis=1, keepdims=True)
            )

            numerical[i, j] = (
                ((y1 - y2) * upstream).sum()
            ) / (2 * eps)

    np.testing.assert_allclose(
        x.grad,
        numerical,
        rtol=1e-4,
        atol=1e-6,
    )