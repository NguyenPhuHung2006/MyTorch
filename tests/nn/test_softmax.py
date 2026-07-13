import numpy as np

from mytorch.tensor import Tensor
from mytorch.nn.modules.activations import Softmax

def test_softmax_forward():
    x = Tensor(np.array([[1.0, 2.0, 3.0]]))
    softmax = Softmax(dim=1)

    y = softmax(x)

    exp = np.exp(x.data - np.max(x.data, axis=1, keepdims=True))
    expected = exp / exp.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(y.data, expected, rtol=1e-6)


def test_softmax_shape():
    x = Tensor(np.random.randn(3, 4, 5))
    softmax = Softmax(dim=-1)

    y = softmax(x)

    assert y.shape == x.shape


def test_softmax_sum_to_one():
    x = Tensor(np.random.randn(8, 10))
    softmax = Softmax(dim=1)

    y = softmax(x)

    np.testing.assert_allclose(
        y.data.sum(axis=1),
        np.ones(8),
        rtol=1e-6,
    )


def test_softmax_dim0():
    x = Tensor(np.random.randn(5, 3))
    softmax = Softmax(dim=0)

    y = softmax(x)

    np.testing.assert_allclose(
        y.data.sum(axis=0),
        np.ones(3),
        rtol=1e-6,
    )


def test_softmax_dim_negative():
    x = Tensor(np.random.randn(2, 3, 4))
    softmax = Softmax(dim=-1)

    y = softmax(x)

    np.testing.assert_allclose(
        y.data.sum(axis=-1),
        np.ones((2, 3)),
        rtol=1e-6,
    )


def test_softmax_backward():
    x = Tensor(
        np.random.randn(3, 4),
        requires_grad=True,
    )

    softmax = Softmax(dim=1)

    y = softmax(x)

    upstream = np.random.randn(*y.shape)

    y.backward(upstream)

    s = y.data
    expected = s * (
        upstream - np.sum(upstream * s, axis=1, keepdims=True)
    )

    np.testing.assert_allclose(
        x.grad,
        expected,
        rtol=1e-6,
    )


def test_softmax_no_parameters():
    softmax = Softmax()

    assert list(softmax.parameters()) == []


def test_softmax_train_eval():
    softmax = Softmax()

    softmax.eval()
    assert softmax.training is False

    softmax.train()
    assert softmax.training is True
    
def test_softmax_gradient_numerical():
    eps = 1e-6

    x_data = np.random.randn(2, 3)

    x = Tensor(x_data.copy(), requires_grad=True)

    softmax = Softmax(dim=1)
    y = softmax(x)

    upstream = np.random.randn(*y.shape)

    y.backward(upstream)

    numerical = np.zeros_like(x_data)

    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            plus = x_data.copy()
            minus = x_data.copy()

            plus[i, j] += eps
            minus[i, j] -= eps

            exp = np.exp(plus - plus.max(axis=1, keepdims=True))
            y1 = exp / exp.sum(axis=1, keepdims=True)

            exp = np.exp(minus - minus.max(axis=1, keepdims=True))
            y2 = exp / exp.sum(axis=1, keepdims=True)

            numerical[i, j] = (
                ((y1 - y2) * upstream).sum()
            ) / (2 * eps)

    np.testing.assert_allclose(
        x.grad,
        numerical,
        rtol=1e-4,
        atol=1e-6,
    )