from mytorch.tensor import Tensor
import numpy as np

def test_T_forward():
    x = Tensor([[1, 2, 3],
                [4, 5, 6]], requires_grad=True)

    y = x.T

    expected = np.array([[1, 4],
                         [2, 5],
                         [3, 6]])

    np.testing.assert_array_equal(y.data, expected)
    
def test_T_backward():
    x = Tensor([[1., 2., 3.],
                [4., 5., 6.]], requires_grad=True)

    y = x.T
    y.backward(np.ones_like(y.data))

    expected_grad = np.ones_like(x.data)

    np.testing.assert_array_equal(x.grad, expected_grad)
    
def test_transpose_default():
    x = Tensor([[1, 2],
                [3, 4]], requires_grad=True)

    y1 = x.T
    y2 = x.transpose()

    np.testing.assert_array_equal(y1.data, y2.data)
    
def test_transpose_dim0_dim2():
    x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)

    y = x.transpose(0, 2)

    expected = np.transpose(x.data, (2, 1, 0))

    np.testing.assert_array_equal(y.data, expected)
    
def test_transpose_negative_dims():
    x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)

    y = x.transpose(-1, -3)

    expected = np.transpose(x.data, (2, 1, 0))

    np.testing.assert_array_equal(y.data, expected)
    
def test_transpose_middle_dims():
    x = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)

    y = x.transpose(1, 2)

    expected = np.transpose(x.data, (0, 2, 1))

    np.testing.assert_array_equal(y.data, expected)
    
def test_transpose_backward():
    x = Tensor(np.arange(24.).reshape(2, 3, 4), requires_grad=True)

    y = x.transpose(0, 2)

    grad = np.random.randn(*y.shape)

    y.backward(grad)

    expected = np.transpose(grad, (2, 1, 0))

    np.testing.assert_allclose(x.grad, expected)
    
def test_transpose_1d_returns_self():
    x = Tensor([1, 2, 3], requires_grad=True)

    y = x.transpose()

    assert y is x
    
def test_transpose_scalar_returns_self():
    x = Tensor(5.0, requires_grad=True)

    y = x.transpose()

    assert y is x
    
def test_T_1d_returns_self():
    x = Tensor([1, 2, 3], requires_grad=True)

    y = x.T

    assert y is x
    
def test_T_scalar_returns_self():
    x = Tensor(5.0, requires_grad=True)

    y = x.T

    assert y is x
    
def test_transpose_shape():
    x = Tensor(np.zeros((2, 3, 4)))

    y = x.transpose(0, 2)

    assert y.shape == (4, 3, 2)
    
def test_double_transpose():
    x = Tensor(np.random.randn(3, 4), requires_grad=True)

    y = x.T.T

    np.testing.assert_allclose(y.data, x.data)
    
def test_double_transpose_backward():
    x = Tensor(np.random.randn(3, 4), requires_grad=True)

    y = x.T.T
    y.backward(np.ones_like(y.data))

    np.testing.assert_allclose(x.grad, np.ones_like(x.data))
    
