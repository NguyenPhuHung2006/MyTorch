import numpy as np

from mytorch.tensor import Tensor
from mytorch.nn.modules.activation import Sigmoid


def test_sigmoid_forward():
    x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    sigmoid = Sigmoid()

    y = sigmoid(x)

    expected = 1 / (1 + np.exp(-x.data))

    np.testing.assert_allclose(y.data, expected, rtol=1e-6)


def test_sigmoid_shape():
    x = Tensor(np.random.randn(3, 4, 5))
    sigmoid = Sigmoid()

    y = sigmoid(x)

    assert y.shape == x.shape


def test_sigmoid_backward():
    x = Tensor(
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        requires_grad=True,
    )

    sigmoid = Sigmoid()
    y = sigmoid(x)

    y.backward(np.ones_like(y.data))

    s = 1 / (1 + np.exp(-x.data))
    expected_grad = s * (1 - s)

    np.testing.assert_allclose(x.grad, expected_grad, rtol=1e-6)


def test_sigmoid_backward_custom_gradient():
    x = Tensor(
        np.array([-1.0, 0.0, 1.0]),
        requires_grad=True,
    )

    sigmoid = Sigmoid()
    y = sigmoid(x)

    upstream = np.array([2.0, 3.0, 4.0])
    y.backward(upstream)

    s = 1 / (1 + np.exp(-x.data))
    expected = upstream * s * (1 - s)

    np.testing.assert_allclose(x.grad, expected, rtol=1e-6)


def test_sigmoid_no_parameters():
    sigmoid = Sigmoid()

    assert list(sigmoid.parameters()) == []


def test_sigmoid_train_eval():
    sigmoid = Sigmoid()

    sigmoid.eval()
    assert sigmoid.training is False

    sigmoid.train()
    assert sigmoid.training is True
    
def test_sigmoid_gradient_numerical():
    eps = 1e-6

    x_data = np.random.randn(5)
    x = Tensor(x_data.copy(), requires_grad=True)

    sigmoid = Sigmoid()
    y = sigmoid(x)
    y.backward(np.ones_like(y.data))

    numerical = np.zeros_like(x_data)

    for i in range(len(x_data)):
        plus = x_data.copy()
        minus = x_data.copy()

        plus[i] += eps
        minus[i] -= eps

        f1 = (1 / (1 + np.exp(-plus))).sum()
        f2 = (1 / (1 + np.exp(-minus))).sum()

        numerical[i] = (f1 - f2) / (2 * eps)

    np.testing.assert_allclose(
        x.grad,
        numerical,
        rtol=1e-4,
        atol=1e-6,
    )