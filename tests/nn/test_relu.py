import numpy as np

from mytorch.tensor import Tensor
from mytorch.nn.modules.activations import ReLU

def test_forward_positive():
    relu = ReLU()

    x = Tensor(np.array([1., 2., 3.]))

    y = relu(x)

    expected = np.array([1., 2., 3.])

    np.testing.assert_allclose(y.data, expected)


def test_forward_negative():
    relu = ReLU()

    x = Tensor(np.array([-1., -2., -3.]))

    y = relu(x)

    expected = np.array([0., 0., 0.])

    np.testing.assert_allclose(y.data, expected)


def test_forward_mixed():
    relu = ReLU()

    x = Tensor(np.array([-2., -1., 0., 1., 2.]))

    y = relu(x)

    expected = np.array([0., 0., 0., 1., 2.])

    np.testing.assert_allclose(y.data, expected)


def test_forward_scalar_positive():
    relu = ReLU()

    x = Tensor(5.)

    y = relu(x)

    np.testing.assert_allclose(y.data, 5.)


def test_forward_scalar_negative():
    relu = ReLU()

    x = Tensor(-5.)

    y = relu(x)

    np.testing.assert_allclose(y.data, 0.)


def test_forward_matrix():
    relu = ReLU()

    x = Tensor(np.array([
        [-2., -1., 0.],
        [1., 2., 3.]
    ]))

    y = relu(x)

    expected = np.array([
        [0., 0., 0.],
        [1., 2., 3.]
    ])

    np.testing.assert_allclose(y.data, expected)


def test_shape_preserved():
    relu = ReLU()

    x = Tensor(np.random.randn(4, 5, 6))

    y = relu(x)

    assert y.shape == x.shape


def test_no_parameters():
    relu = ReLU()

    params = list(relu.parameters())

    assert params == []


def test_named_parameters():
    relu = ReLU()

    params = dict(relu.named_parameters())

    assert params == {}


def test_train():
    relu = ReLU()

    relu.train()

    assert relu.training


def test_eval():
    relu = ReLU()

    relu.eval()

    assert not relu.training


def test_zero_grad():
    relu = ReLU()

    # Should simply do nothing without raising an exception.
    relu.zero_grad()


def test_multiple_calls():
    relu = ReLU()

    x = Tensor(np.array([-1., 2., -3., 4.]))

    y1 = relu(x)
    y2 = relu(x)

    np.testing.assert_allclose(y1.data, y2.data)


def test_batch():
    relu = ReLU()

    x = Tensor(np.random.randn(32, 128))

    y = relu(x)

    assert y.shape == (32, 128)

    np.testing.assert_array_less(-1e-12, y.data)