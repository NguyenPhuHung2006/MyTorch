import numpy as np

from mytorch.nn import Parameter
from mytorch.optim import Adam


def test_adam_single_parameter():
    p = Parameter(np.array([1.0]))
    p.grad = np.array([2.0])

    betas = (0.9, 0.999)

    optimizer = Adam(
        [p],
        lr=0.1,
        betas=betas,
        eps=1e-8,
    )

    optimizer.step()

    beta1, beta2 = betas

    # First Adam update
    m = (1 - beta1) * 2.0
    v = (1 - beta2) * (2.0 ** 2)

    m_hat = m / (1 - beta1)
    v_hat = v / (1 - beta2)

    expected = 1.0 - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)

    np.testing.assert_allclose(p.data, [expected])

def test_skip_none_gradient():
    p = Parameter(np.array([5.0]))
    p.grad = None

    optimizer = Adam([p], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(p.data, [5.0])


def test_zero_gradient():
    p = Parameter(np.array([5.0]))
    p.grad = np.array([0.0])

    optimizer = Adam([p], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(p.data, [5.0])


def test_multiple_parameters():
    p1 = Parameter(np.array([1.0]))
    p2 = Parameter(np.array([2.0]))

    p1.grad = np.array([2.0])
    p2.grad = np.array([-3.0])

    optimizer = Adam([p1, p2], lr=0.1)
    optimizer.step()

    assert np.isfinite(p1.data).all()
    assert np.isfinite(p2.data).all()

    assert not np.array_equal(p1.data, np.array([1.0]))
    assert not np.array_equal(p2.data, np.array([2.0]))


def test_vector_parameter():
    p = Parameter(np.array([1.0, 2.0, 3.0]))
    p.grad = np.array([1.0, 2.0, 3.0])

    optimizer = Adam([p], lr=0.01)
    optimizer.step()

    assert p.data.shape == (3,)
    assert np.isfinite(p.data).all()


def test_constant_gradient_decreases_parameter():
    p = Parameter(np.array([1.0]))

    optimizer = Adam([p], lr=0.01)

    previous = p.data.copy()

    for _ in range(20):
        p.grad = np.array([1.0])
        optimizer.step()

        assert p.data < previous
        previous = p.data.copy()


def test_gradient_direction():
    p = Parameter(np.array([0.0]))

    optimizer = Adam([p], lr=0.1)

    p.grad = np.array([1.0])
    optimizer.step()

    first = p.data.copy()

    p.grad = np.array([-1.0])
    optimizer.step()

    second = p.data.copy()

    # Parameter should move back toward zero,
    # but should remain finite.
    assert np.isfinite(second).all()
    assert second > first


def test_no_nan_after_many_steps():
    p = Parameter(np.array([1.0]))

    optimizer = Adam([p], lr=0.001)

    for i in range(1000):
        p.grad = np.array([np.sin(i)])
        optimizer.step()

    assert np.isfinite(p.data).all()


def test_small_gradient():
    p = Parameter(np.array([1.0]))
    p.grad = np.array([1e-12])

    optimizer = Adam([p], lr=0.01)
    optimizer.step()

    assert np.isfinite(p.data).all()


def test_large_gradient():
    p = Parameter(np.array([1.0]))
    p.grad = np.array([1e6])

    optimizer = Adam([p], lr=0.01)
    optimizer.step()

    assert np.isfinite(p.data).all()