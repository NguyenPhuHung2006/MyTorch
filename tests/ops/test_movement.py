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
        
import numpy as np
import pytest

from mytorch import Tensor


# ============================================================
# Forward tests
# ============================================================

def test_expand_first_dimension():
    x = Tensor(
        np.array([[1, 2, 3]])
    )

    # (1, 3) -> (4, 3)
    y = x.expand(4, 3)

    expected = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ])

    assert y.shape == (4, 3)
    np.testing.assert_array_equal(y.data, expected)


def test_expand_middle_dimension():
    x = Tensor(
        np.array([
            [[1, 2, 3]],
            [[4, 5, 6]],
        ])
    )

    # (2, 1, 3) -> (2, 4, 3)
    y = x.expand(2, 4, 3)

    expected = np.array([
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ],
        [
            [4, 5, 6],
            [4, 5, 6],
            [4, 5, 6],
            [4, 5, 6],
        ],
    ])

    assert y.shape == (2, 4, 3)
    np.testing.assert_array_equal(y.data, expected)


def test_expand_multiple_dimensions():
    x = Tensor(
        np.array([[[5]]])
    )

    # (1, 1, 1) -> (2, 3, 4)
    y = x.expand(2, 3, 4)

    expected = np.full(
        (2, 3, 4),
        5,
    )

    assert y.shape == (2, 3, 4)
    np.testing.assert_array_equal(y.data, expected)


def test_expand_add_leading_dimension():
    x = Tensor(
        np.array([1, 2, 3])
    )

    # (3,) -> (4, 3)
    y = x.expand(4, 3)

    expected = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ])

    assert y.shape == (4, 3)
    np.testing.assert_array_equal(y.data, expected)


def test_expand_same_shape():
    x = Tensor(
        np.array([
            [1, 2],
            [3, 4],
        ])
    )

    y = x.expand(2, 2)

    assert y.shape == (2, 2)
    np.testing.assert_array_equal(y.data, x.data)


# ============================================================
# Invalid expansion tests
# ============================================================

def test_expand_invalid_dimension():
    x = Tensor(
        np.array([
            [1, 2, 3],
        ])
    )

    # (1, 3) cannot become (2, 4)
    with pytest.raises(ValueError):
        x.expand(2, 4)


def test_expand_non_singleton_dimension():
    x = Tensor(
        np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
    )

    # Dimension 0 has size 2 and cannot become 3
    with pytest.raises(ValueError):
        x.expand(3, 3)


def test_expand_too_few_dimensions():
    x = Tensor(
        np.ones((2, 3))
    )

    # Cannot remove dimensions with expand
    with pytest.raises(ValueError):
        x.expand(3)


# ============================================================
# Backward tests
# ============================================================

def test_expand_backward_first_dimension():
    x = Tensor(
        np.array([[1.0, 2.0, 3.0]]),
        requires_grad=True,
    )

    y = x.expand(4, 3)

    loss = y.sum()
    loss.backward()

    # Each value was expanded 4 times.
    expected_grad = np.array([
        [4.0, 4.0, 4.0],
    ])

    np.testing.assert_allclose(
        x.grad,
        expected_grad,
    )


def test_expand_backward_middle_dimension():
    x = Tensor(
        np.array([
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
        ]),
        requires_grad=True,
    )

    # (2, 1, 3) -> (2, 4, 3)
    y = x.expand(2, 4, 3)

    loss = y.sum()
    loss.backward()

    expected_grad = np.array([
        [[4.0, 4.0, 4.0]],
        [[4.0, 4.0, 4.0]],
    ])

    np.testing.assert_allclose(
        x.grad,
        expected_grad,
    )


def test_expand_backward_multiple_dimensions():
    x = Tensor(
        np.array([[[1.0]]]),
        requires_grad=True,
    )

    # (1, 1, 1) -> (2, 3, 4)
    y = x.expand(2, 3, 4)

    loss = y.sum()
    loss.backward()

    # The value appears 2 * 3 * 4 = 24 times.
    expected_grad = np.array([
        [[24.0]],
    ])

    np.testing.assert_allclose(
        x.grad,
        expected_grad,
    )


def test_expand_backward_leading_dimension():
    x = Tensor(
        np.array([1.0, 2.0, 3.0]),
        requires_grad=True,
    )

    # (3,) -> (5, 3)
    y = x.expand(5, 3)

    loss = y.sum()
    loss.backward()

    expected_grad = np.array([
        5.0,
        5.0,
        5.0,
    ])

    np.testing.assert_allclose(
        x.grad,
        expected_grad,
    )


# ============================================================
# Non-uniform gradient test
# ============================================================

def test_expand_backward_non_uniform_gradient():
    x = Tensor(
        np.array([[1.0, 2.0, 3.0]]),
        requires_grad=True,
    )

    y = x.expand(3, 3)

    # Give every output element a different gradient.
    weights = Tensor(
        np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
    )

    loss = (y * weights).sum()
    loss.backward()

    # Gradient must be summed along the expanded
    # first dimension:
    #
    # [1 + 4 + 7,
    #  2 + 5 + 8,
    #  3 + 6 + 9]
    expected_grad = np.array([
        [12.0, 15.0, 18.0],
    ])

    np.testing.assert_allclose(
        x.grad,
        expected_grad,
    )


# ============================================================
# CLS-token-like tests
# ============================================================

def test_expand_cls_token():
    d_model = 8
    batch_size = 16

    x = Tensor(
        np.random.randn(1, 1, d_model),
        requires_grad=True,
    )

    y = x.expand(
        batch_size,
        1,
        d_model,
    )

    assert y.shape == (
        batch_size,
        1,
        d_model,
    )

    # Every batch element should contain
    # the same CLS token.
    for i in range(batch_size):
        np.testing.assert_array_equal(
            y.data[i],
            x.data[0],
        )


def test_expand_cls_token_backward():
    d_model = 8
    batch_size = 16

    x = Tensor(
        np.random.randn(1, 1, d_model),
        requires_grad=True,
    )

    y = x.expand(
        batch_size,
        1,
        d_model,
    )

    loss = y.sum()
    loss.backward()

    # The same CLS parameter is shared by every
    # sample, so gradients accumulate over batch.
    expected_grad = np.full(
        (1, 1, d_model),
        batch_size,
        dtype=float,
    )

    np.testing.assert_allclose(
        x.grad,
        expected_grad,
    )