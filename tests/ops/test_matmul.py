import numpy as np
from mytorch.tensor import Tensor

def test_matmul_matrix_matrix():
    a = Tensor([[1, 2],
                [3, 4]])
    b = Tensor([[5, 6],
                [7, 8]])

    c = a @ b

    np.testing.assert_array_equal(
        c.data,
        np.array([[19, 22],
                  [43, 50]])
    )


def test_matmul_matrix_vector():
    a = Tensor([[1, 2, 3],
                [4, 5, 6]])
    b = Tensor([10, 20, 30])

    c = a @ b

    np.testing.assert_array_equal(
        c.data,
        np.array([140, 320])
    )


def test_matmul_vector_matrix():
    a = Tensor([1, 2, 3])
    b = Tensor([[1, 2],
                [3, 4],
                [5, 6]])

    c = a @ b

    np.testing.assert_array_equal(
        c.data,
        np.array([22, 28])
    )


def test_matmul_vector_vector():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])

    c = a @ b

    assert c.data == 32
    
def test_batched_matmul():
    a = Tensor(np.arange(24).reshape(2, 3, 4))
    b = Tensor(np.arange(40).reshape(2, 4, 5))

    c = a @ b

    expected = np.matmul(a.data, b.data)

    np.testing.assert_array_equal(c.data, expected)
    
def test_broadcast_batch_dimension():
    a = Tensor(np.random.randn(5, 2, 3))
    b = Tensor(np.random.randn(3, 4))

    c = a @ b

    expected = np.matmul(a.data, b.data)

    np.testing.assert_allclose(c.data, expected)
    
def test_broadcast_multiple_batch_dimensions():
    a = Tensor(np.random.randn(2, 5, 3, 4))
    b = Tensor(np.random.randn(5, 4, 6))

    c = a @ b

    expected = np.matmul(a.data, b.data)

    np.testing.assert_allclose(c.data, expected)
    
def test_row_column():
    a = Tensor([[1, 2, 3]])
    b = Tensor([[4],
                [5],
                [6]])

    c = a @ b

    np.testing.assert_array_equal(
        c.data,
        np.array([[32]])
    )
    
import pytest

def test_invalid_shape():
    a = Tensor(np.random.randn(2, 3))
    b = Tensor(np.random.randn(4, 5))

    with pytest.raises(ValueError):
        _ = a @ b
        
def test_scalar_not_allowed():
    a = Tensor(3)
    b = Tensor(4)

    with pytest.raises(ValueError):
        _ = a @ b
        
def test_empty_dimension():
    a = Tensor(np.empty((0, 3)))
    b = Tensor(np.empty((3, 2)))

    c = a @ b

    assert c.data.shape == (0, 2)
    
def test_random_against_numpy():
    rng = np.random.default_rng(42)

    shapes = [
        ((2, 3), (3, 4)),
        ((5, 2, 3), (3, 4)),
        ((2, 5, 3, 4), (5, 4, 6)),
        ((3,), (3,)),
        ((3,), (3, 4)),
        ((4, 3), (3,))
    ]

    for sa, sb in shapes:
        a = rng.standard_normal(sa)
        b = rng.standard_normal(sb)

        ta = Tensor(a)
        tb = Tensor(b)

        out = ta @ tb
        expected = np.matmul(a, b)

        np.testing.assert_allclose(out.data, expected)
        
def test_matmul_vector_vector_backward():
    x = Tensor([1., 2., 3.], requires_grad=True)
    y = Tensor([4., 5., 6.], requires_grad=True)

    z = x @ y
    z.backward()

    np.testing.assert_allclose(x.grad, [4., 5., 6.])
    np.testing.assert_allclose(y.grad, [1., 2., 3.])
    
def test_matmul_matrix_vector_backward():
    x = Tensor([[1., 2., 3.],
                [4., 5., 6.]], requires_grad=True)

    y = Tensor([1., 2., 3.], requires_grad=True)

    z = (x @ y).sum()
    z.backward()

    np.testing.assert_allclose(
        x.grad,
        np.array([[1., 2., 3.],
                  [1., 2., 3.]])
    )

    np.testing.assert_allclose(
        y.grad,
        np.array([5., 7., 9.])
    )
    
def test_matmul_vector_matrix_backward():
    x = Tensor([1., 2., 3.], requires_grad=True)

    y = Tensor([[1., 2.],
                [3., 4.],
                [5., 6.]], requires_grad=True)

    z = (x @ y).sum()
    z.backward()

    np.testing.assert_allclose(
        x.grad,
        np.array([3., 7., 11.])
    )

    np.testing.assert_allclose(
        y.grad,
        np.array([[1., 1.],
                  [2., 2.],
                  [3., 3.]])
    )
    
def test_matmul_rhs_not_tensor():
    x = Tensor([[1., 2.],
                [3., 4.]], requires_grad=True)

    y = np.array([[5., 6.],
                  [7., 8.]])

    z = (x @ y).sum()

    # Should not crash.
    z.backward()