import numpy as np
from mytorch.tensor import Tensor

def test_add_broadcast():
    x = Tensor(
        [[1., 2., 3.],
         [4., 5., 6.]],
        requires_grad=True
    )

    y = Tensor(
        [10., 20., 30.],
        requires_grad=True
    )

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