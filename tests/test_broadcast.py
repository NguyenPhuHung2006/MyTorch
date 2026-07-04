from mytorch.tensor import Tensor
import numpy as np

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