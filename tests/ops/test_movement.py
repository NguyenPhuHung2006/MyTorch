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
    

def test_reshape_shape():
    x = Tensor(np.arange(24).reshape(2, 3, 4))

    y = x.reshape((6, 4))

    assert y.shape == (6, 4)


def test_reshape_values():
    x = Tensor(np.arange(12))

    y = x.reshape((3, 4))

    np.testing.assert_array_equal(
        y.data,
        np.arange(12).reshape(3, 4)
    )


def test_reshape_backward():
    x = Tensor(np.random.randn(2, 3), requires_grad=True)

    y = x.reshape((3, 2))
    loss = y.sum()

    loss.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.ones((2, 3))
    )
    
import numpy as np

from mytorch import Tensor


def test_flatten_default():
    x = Tensor(np.zeros((2, 3, 4)))

    y = x.flatten()

    assert y.shape == (24,)


def test_flatten_start_dim():
    x = Tensor(np.zeros((2, 3, 4)))

    y = x.flatten(start_dim=1)

    assert y.shape == (2, 12)


def test_flatten_middle_dims():
    x = Tensor(np.zeros((3, 2, 4, 5)))

    y = x.flatten(1, 2)

    assert y.shape == (3, 8, 5)


def test_flatten_negative_end_dim():
    x = Tensor(np.zeros((3, 2, 4, 5)))

    y = x.flatten(1, -1)

    assert y.shape == (3, 40)


def test_flatten_negative_start_dim():
    x = Tensor(np.zeros((2, 3, 4)))

    y = x.flatten(-2, -1)

    assert y.shape == (2, 12)


def test_flatten_values():
    x = Tensor(np.arange(24).reshape(2, 3, 4))

    y = x.flatten(1)

    np.testing.assert_array_equal(
        y.data,
        np.arange(24).reshape(2, 12)
    )


def test_flatten_backward():
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)

    y = x.flatten(1)

    loss = y.sum()

    loss.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.ones((2, 3, 4))
    )
    
import pytest

def test_flatten_invalid_start_dim():
    x = Tensor(np.zeros((2, 3)))

    with pytest.raises(IndexError):
        x.flatten(3)


def test_flatten_invalid_end_dim():
    x = Tensor(np.zeros((2, 3)))

    with pytest.raises(IndexError):
        x.flatten(0, 3)


def test_flatten_start_after_end():
    x = Tensor(np.zeros((2, 3, 4)))

    with pytest.raises(ValueError):
        x.flatten(2, 1)
        
def test_reshape_preserves_order():
    x = Tensor(np.arange(24).reshape(2, 3, 4))

    y = x.reshape((4, 6))

    np.testing.assert_array_equal(
        y.data.ravel(),
        x.data.ravel()
    )

from mytorch import Tensor, cat, stack

# ==========================
# Cat
# ==========================

def test_cat_forward_axis0():
    a = Tensor(np.random.randn(2, 3))
    b = Tensor(np.random.randn(4, 3))

    y = cat([a, b], axis=0)

    expected = np.concatenate([a.data, b.data], axis=0)

    assert y.shape == expected.shape
    np.testing.assert_allclose(y.numpy(), expected)


def test_cat_forward_axis1():
    a = Tensor(np.random.randn(2, 3))
    b = Tensor(np.random.randn(2, 5))

    y = cat([a, b], axis=1)

    expected = np.concatenate([a.data, b.data], axis=1)

    assert y.shape == expected.shape
    np.testing.assert_allclose(y.numpy(), expected)


def test_cat_backward_two_inputs():
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    b = Tensor(np.random.randn(4, 3), requires_grad=True)

    y = cat([a, b], axis=0)
    loss = y.sum()
    loss.backward()

    np.testing.assert_allclose(a.grad, np.ones_like(a.data))
    np.testing.assert_allclose(b.grad, np.ones_like(b.data))


def test_cat_backward_three_inputs():
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    b = Tensor(np.random.randn(1, 3), requires_grad=True)
    c = Tensor(np.random.randn(4, 3), requires_grad=True)

    y = cat([a, b, c], axis=0)
    loss = y.sum()
    loss.backward()

    np.testing.assert_allclose(a.grad, np.ones_like(a.data))
    np.testing.assert_allclose(b.grad, np.ones_like(b.data))
    np.testing.assert_allclose(c.grad, np.ones_like(c.data))


def test_cat_requires_grad():
    a = Tensor(np.random.randn(2, 3))
    b = Tensor(np.random.randn(2, 3), requires_grad=True)

    y = cat([a, b], axis=0)

    assert y.requires_grad


# ==========================
# Stack
# ==========================

def test_stack_forward_axis0():
    a = Tensor(np.random.randn(2, 3))
    b = Tensor(np.random.randn(2, 3))

    y = stack([a, b], axis=0)

    expected = np.stack([a.data, b.data], axis=0)

    assert y.shape == expected.shape
    np.testing.assert_allclose(y.numpy(), expected)


def test_stack_forward_axis1():
    a = Tensor(np.random.randn(2, 3))
    b = Tensor(np.random.randn(2, 3))

    y = stack([a, b], axis=1)

    expected = np.stack([a.data, b.data], axis=1)

    assert y.shape == expected.shape
    np.testing.assert_allclose(y.numpy(), expected)


def test_stack_backward_two_inputs():
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    b = Tensor(np.random.randn(2, 3), requires_grad=True)

    y = stack([a, b], axis=0)
    loss = y.sum()
    loss.backward()

    np.testing.assert_allclose(a.grad, np.ones_like(a.data))
    np.testing.assert_allclose(b.grad, np.ones_like(b.data))


def test_stack_backward_three_inputs():
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    b = Tensor(np.random.randn(2, 3), requires_grad=True)
    c = Tensor(np.random.randn(2, 3), requires_grad=True)

    y = stack([a, b, c], axis=1)
    loss = y.sum()
    loss.backward()

    np.testing.assert_allclose(a.grad, np.ones_like(a.data))
    np.testing.assert_allclose(b.grad, np.ones_like(b.data))
    np.testing.assert_allclose(c.grad, np.ones_like(c.data))


def test_stack_requires_grad():
    a = Tensor(np.random.randn(2, 3))
    b = Tensor(np.random.randn(2, 3), requires_grad=True)

    y = stack([a, b], axis=0)

    assert y.requires_grad


# ==========================
# Errors
# ==========================

def test_cat_shape_mismatch():
    a = Tensor(np.random.randn(2, 3))
    b = Tensor(np.random.randn(4, 4))

    with np.testing.assert_raises(ValueError):
        cat([a, b], axis=0)


def test_stack_shape_mismatch():
    a = Tensor(np.random.randn(2, 3))
    b = Tensor(np.random.randn(3, 3))

    with np.testing.assert_raises(ValueError):
        stack([a, b], axis=0)