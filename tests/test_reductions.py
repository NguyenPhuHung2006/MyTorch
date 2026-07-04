import numpy as np
from mytorch.tensor import Tensor


# =========================
# Sum
# =========================

def test_sum():
    x = Tensor(
        np.array([[1., 2.],
                  [3., 4.]]),
        requires_grad=True
    )

    y = x.sum()

    assert y.data == 10.

    y.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.ones((2, 2))
    )


def test_sum_axis0():
    x = Tensor(
        np.array([[1., 2., 3.],
                  [4., 5., 6.]]),
        requires_grad=True
    )

    y = x.sum(axis=0)

    np.testing.assert_array_equal(
        y.data,
        np.array([5., 7., 9.])
    )

    y.backward(np.ones_like(y.data))

    np.testing.assert_array_equal(
        x.grad,
        np.ones((2, 3))
    )


def test_sum_axis1():
    x = Tensor(
        np.array([[1., 2., 3.],
                  [4., 5., 6.]]),
        requires_grad=True
    )

    y = x.sum(axis=1)

    np.testing.assert_array_equal(
        y.data,
        np.array([6., 15.])
    )

    y.backward(np.array([1., 2.]))

    np.testing.assert_array_equal(
        x.grad,
        np.array([
            [1., 1., 1.],
            [2., 2., 2.]
        ])
    )


def test_sum_keepdims():
    x = Tensor(
        np.ones((2, 3)),
        requires_grad=True
    )

    y = x.sum(axis=1, keepdims=True)

    assert y.data.shape == (2, 1)

    y.backward(np.array([[1.], [2.]]))

    np.testing.assert_array_equal(
        x.grad,
        np.array([
            [1., 1., 1.],
            [2., 2., 2.]
        ])
    )


# =========================
# Mean
# =========================

def test_mean():
    x = Tensor(
        np.array([[1., 2.],
                  [3., 4.]]),
        requires_grad=True
    )

    y = x.mean()

    assert y.data == 2.5

    y.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.full((2, 2), 0.25)
    )


def test_mean_axis0():
    x = Tensor(
        np.ones((2, 3)),
        requires_grad=True
    )

    y = x.mean(axis=0)

    np.testing.assert_array_equal(
        y.data,
        np.ones(3)
    )

    y.backward(np.ones_like(y.data))

    np.testing.assert_array_equal(
        x.grad,
        np.full((2, 3), 0.5)
    )


def test_mean_axis1():
    x = Tensor(
        np.ones((2, 3)),
        requires_grad=True
    )

    y = x.mean(axis=1)

    np.testing.assert_array_equal(
        y.data,
        np.ones(2)
    )

    y.backward(np.array([1., 2.]))

    np.testing.assert_array_equal(
        x.grad,
        np.array([
            [1/3, 1/3, 1/3],
            [2/3, 2/3, 2/3]
        ])
    )


# =========================
# Max
# =========================

def test_max():
    x = Tensor(
        np.array([1., 3., 2.]),
        requires_grad=True
    )

    y = x.max()

    assert y.data == 3.

    y.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.array([0., 1., 0.])
    )


def test_max_duplicate():
    x = Tensor(
        np.array([3., 3.]),
        requires_grad=True
    )

    y = x.max()
    y.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.array([0.5, 0.5])
    )


def test_max_axis0():
    x = Tensor(
        np.array([
            [1., 5., 3.],
            [4., 2., 6.]
        ]),
        requires_grad=True
    )

    y = x.max(axis=0)

    np.testing.assert_array_equal(
        y.data,
        np.array([4., 5., 6.])
    )

    y.backward(np.ones_like(y.data))

    np.testing.assert_array_equal(
        x.grad,
        np.array([
            [0., 1., 0.],
            [1., 0., 1.]
        ])
    )


# =========================
# Min
# =========================

def test_min():
    x = Tensor(
        np.array([1., 3., 2.]),
        requires_grad=True
    )

    y = x.min()

    assert y.data == 1.

    y.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.array([1., 0., 0.])
    )


def test_min_duplicate():
    x = Tensor(
        np.array([1., 1.]),
        requires_grad=True
    )

    y = x.min()
    y.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.array([0.5, 0.5])
    )


def test_min_axis0():
    x = Tensor(
        np.array([
            [1., 5., 3.],
            [4., 2., 6.]
        ]),
        requires_grad=True
    )

    y = x.min(axis=0)

    np.testing.assert_array_equal(
        y.data,
        np.array([1., 2., 3.])
    )

    y.backward(np.ones_like(y.data))

    np.testing.assert_array_equal(
        x.grad,
        np.array([
            [1., 0., 1.],
            [0., 1., 0.]
        ])
    )