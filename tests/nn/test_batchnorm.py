import numpy as np
import pytest

from mytorch import Tensor
from mytorch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

def test_batchnorm1d_output_shape():
    bn = BatchNorm1d(5)
    x = Tensor(np.random.randn(8, 5))

    y = bn(x)

    assert y.shape == (8, 5)


def test_batchnorm1d_sequence_output_shape():
    bn = BatchNorm1d(5)
    x = Tensor(np.random.randn(8, 5, 10))

    y = bn(x)

    assert y.shape == (8, 5, 10)


def test_batchnorm2d_output_shape():
    bn = BatchNorm2d(4)
    x = Tensor(np.random.randn(2, 4, 8, 8))

    y = bn(x)

    assert y.shape == x.shape


def test_batchnorm3d_output_shape():
    bn = BatchNorm3d(4)
    x = Tensor(np.random.randn(2, 4, 6, 8, 8))

    y = bn(x)

    assert y.shape == x.shape


def test_batchnorm1d_zero_mean_unit_variance():
    bn = BatchNorm1d(6)
    bn.train()

    x = Tensor(np.random.randn(64, 6))

    y = bn(x)

    np.testing.assert_allclose(
        y.mean(axis=0).numpy(),
        np.zeros(6),
        atol=1e-5,
    )
    
    var = ((y - y.mean(axis=0)) ** 2).mean(axis=0).numpy()

    np.testing.assert_allclose(
        var,
        np.ones(6),
        atol=1e-4,
    )


def test_batchnorm_updates_running_mean():
    bn = BatchNorm1d(5)

    old = bn.running_mean.copy()

    x = Tensor(np.random.randn(32, 5))

    bn.train()
    bn(x)

    assert not np.allclose(old, bn.running_mean)


def test_batchnorm_updates_running_var():
    bn = BatchNorm1d(5)

    old = bn.running_var.copy()

    x = Tensor(np.random.randn(32, 5))

    bn.train()
    bn(x)

    assert not np.allclose(old, bn.running_var)


def test_batchnorm_eval_does_not_update_running_stats():
    bn = BatchNorm1d(5)

    bn.train()
    bn(Tensor(np.random.randn(32, 5)))

    running_mean = bn.running_mean.copy()
    running_var = bn.running_var.copy()

    bn.eval()
    bn(Tensor(np.random.randn(32, 5)))

    np.testing.assert_array_equal(running_mean, bn.running_mean)
    np.testing.assert_array_equal(running_var, bn.running_var)


def test_batchnorm_affine_false():
    bn = BatchNorm1d(5, affine=False)

    assert bn.weight is None
    assert bn.bias is None

    x = Tensor(np.random.randn(8, 5))

    y = bn(x)

    assert y.shape == x.shape


def test_batchnorm_track_running_stats_false():
    bn = BatchNorm1d(5, track_running_stats=False)

    assert bn.running_mean is None
    assert bn.running_var is None

    x = Tensor(np.random.randn(16, 5))

    y = bn(x)

    assert y.shape == x.shape


def test_batchnorm1d_invalid_dimension():
    bn = BatchNorm1d(5)

    x = Tensor(np.random.randn(2, 5, 4, 3))

    with pytest.raises(ValueError):
        bn(x)


def test_batchnorm2d_invalid_dimension():
    bn = BatchNorm2d(5)

    x = Tensor(np.random.randn(4, 5))

    with pytest.raises(ValueError):
        bn(x)


def test_batchnorm3d_invalid_dimension():
    bn = BatchNorm3d(5)

    x = Tensor(np.random.randn(2, 5, 8, 8))

    with pytest.raises(ValueError):
        bn(x)


def test_batchnorm_wrong_num_features():
    bn = BatchNorm2d(8)

    x = Tensor(np.random.randn(4, 6, 10, 10))

    with pytest.raises(ValueError):
        bn(x)


# ----------------------------------------------------------------------
# These tests demonstrate why mean/var must remain Tensors.
# They require your autograd engine to support backward().
# ----------------------------------------------------------------------

def test_mean_backward():
    x = Tensor(np.array([1., 2., 3.]), requires_grad=True)

    y = x.mean()
    y.backward()

    np.testing.assert_allclose(
        x.grad,
        np.array([1/3, 1/3, 1/3]),
        atol=1e-6,
    )


def test_variance_backward():
    x = Tensor(np.array([1., 2., 3.]), requires_grad=True)

    v = ((x - x.mean()) ** 2).mean()
    v.backward()

    expected = np.array([-2/3, 0., 2/3])

    np.testing.assert_allclose(
        x.grad,
        expected,
        atol=1e-6,
    )