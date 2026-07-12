from mytorch.tensor import Tensor
import numpy as np

def test_leaf():
    x = Tensor(3.0, requires_grad=True)
    x.backward()

    assert x.grad == 1.0
    
def test_add():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    z = x + y
    z.backward()

    assert x.grad == 1.0
    assert y.grad == 1.0
    
def test_sub():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    z = x - y
    z.backward()

    assert x.grad == 1.0
    assert y.grad == -1.0
    
def test_mul():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    z = x * y
    z.backward()

    assert x.grad == 3.0
    assert y.grad == 2.0
    
def test_chain():
    x = Tensor(2.0, requires_grad=True)
    y = x * x

    y.backward()

    assert x.grad == 4.0
    
def test_chain2():
    x = Tensor(2.0, requires_grad=True)

    y = x * x
    z = y * x

    z.backward()

    assert x.grad == 12.0
    
def test_accumulation():
    x = Tensor(2.0, requires_grad=True)

    y = x + x
    y.backward()

    assert x.grad == 2.0
    
def test_diamond():
    x = Tensor(2.0, requires_grad=True)

    y = x * 2
    z = x * 3
    w = y + z

    w.backward()

    assert x.grad == 5.0
    
def test_shared_node():
    x = Tensor(2.0, requires_grad=True)

    y = x * x
    z = y + y

    z.backward()

    assert x.grad == 8.0
    
def test_constant():
    x = Tensor(2.0, requires_grad=True)

    y = x + 3
    y.backward()

    assert x.grad == 1.0
    
def test_constant_mul():
    x = Tensor(2.0, requires_grad=True)

    y = x * 5
    y.backward()

    assert x.grad == 5.0
    
def test_no_grad():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=False)

    z = x * y
    z.backward()

    assert x.grad == 3.0
    assert y.grad is None
    
def test_multiple_backward():
    x = Tensor(2.0, requires_grad=True)

    y = x * x
    y.backward()
    y.backward()

    assert x.grad == 8.0
    
def numerical_grad(f, x, eps=1e-6):
    x1 = x + eps
    x2 = x - eps
    return (f(x1) - f(x2)) / (2 * eps)

def test_gradcheck():
    x = Tensor(2.0, requires_grad=True)

    y = x * x * x
    y.backward()

    expected = numerical_grad(
        lambda v: v * v * v,
        2.0
    )

    assert np.allclose(x.grad, expected)

def test_broadcast_add():
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    y = Tensor(np.ones((3,)), requires_grad=True)

    z = x + y
    z.sum().backward()

    np.testing.assert_array_equal(
        x.grad,
        np.ones((2, 3))
    )

    np.testing.assert_array_equal(
        y.grad,
        np.array([2., 2., 2.])
    )
    
def test_broadcast_mul():
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    y = Tensor(np.array([1., 2., 3.]), requires_grad=True)

    z = x * y
    z.sum().backward()

    np.testing.assert_array_equal(
        x.grad,
        np.array([
            [1., 2., 3.],
            [1., 2., 3.]
        ])
    )

    np.testing.assert_array_equal(
        y.grad,
        np.array([2., 2., 2.])
    )