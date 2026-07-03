import numpy as np
from mytorch.tensor import Tensor

def test_add_backward():
    x = Tensor([1., 2., 3.], requires_grad=True)
    y = Tensor([4., 5., 6.], requires_grad=True)

    z = x + y
    z.sum().backward()

    np.testing.assert_array_equal(
        x.grad,
        np.array([1., 1., 1.])
    )

    np.testing.assert_array_equal(
        y.grad,
        np.array([1., 1., 1.])
    )