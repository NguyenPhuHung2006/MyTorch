import numpy as np

from mytorch.nn import Parameter
from mytorch.optim import RMSProp


def test_rmsprop_single_parameter():
    p = Parameter(np.array([1.0]))
    p.grad = np.array([2.0])

    optimizer = RMSProp([p], lr=0.1, alpha=0.9, eps=1e-8)
    optimizer.step()

    square_avg = 0.9 * 0 + 0.1 * (2.0 ** 2)
    expected = 1.0 - 0.1 * 2.0 / (np.sqrt(square_avg) + 1e-8)

    np.testing.assert_allclose(p.data, [expected])


def test_rmsprop_multiple_parameters():
    p1 = Parameter(np.array([1.0]))
    p2 = Parameter(np.array([2.0]))

    p1.grad = np.array([2.0])
    p2.grad = np.array([-3.0])

    optimizer = RMSProp([p1, p2], lr=0.1, alpha=0.9)

    optimizer.step()

    avg1 = 0.1 * 4
    avg2 = 0.1 * 9

    expected1 = 1.0 - 0.1 * 2 / np.sqrt(avg1)
    expected2 = 2.0 - 0.1 * (-3) / np.sqrt(avg2)

    np.testing.assert_allclose(p1.data, [expected1])
    np.testing.assert_allclose(p2.data, [expected2])


def test_skip_none_gradient():
    p = Parameter(np.array([5.0]))
    p.grad = None

    optimizer = RMSProp([p], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(p.data, [5.0])


def test_zero_gradient():
    p = Parameter(np.array([5.0]))
    p.grad = np.array([0.0])

    optimizer = RMSProp([p], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(p.data, [5.0])


def test_rmsprop_accumulates_square_average():
    p = Parameter(np.array([1.0]))

    optimizer = RMSProp([p], lr=0.1, alpha=0.9)

    p.grad = np.array([1.0])
    optimizer.step()

    first = p.data.copy()

    p.grad = np.array([1.0])
    optimizer.step()

    second = p.data.copy()

    avg1 = 0.1

    avg2 = 0.9 * avg1 + 0.1

    expected_first = 1.0 - 0.1 / np.sqrt(avg1)

    expected_second = expected_first - 0.1 / np.sqrt(avg2)

    np.testing.assert_allclose(first, [expected_first])
    np.testing.assert_allclose(second, [expected_second])


def test_large_gradient_is_scaled():
    p = Parameter(np.array([1.0]))
    p.grad = np.array([1000.0])

    optimizer = RMSProp([p], lr=0.1)
    optimizer.step()

    # Should not explode
    assert np.isfinite(p.data).all()


def test_small_gradient():
    p = Parameter(np.array([1.0]))
    p.grad = np.array([1e-6])

    optimizer = RMSProp([p], lr=0.1)
    optimizer.step()

    assert np.isfinite(p.data).all()


def test_vector_parameter():
    p = Parameter(np.array([1.0, 2.0, 3.0]))
    p.grad = np.array([1.0, 2.0, 3.0])

    optimizer = RMSProp([p], lr=0.1, alpha=0.9)
    optimizer.step()

    avg = 0.1 * np.square([1.0, 2.0, 3.0])

    expected = np.array([1.0, 2.0, 3.0]) - (
        0.1 * np.array([1.0, 2.0, 3.0]) /
        np.sqrt(avg)
    )

    np.testing.assert_allclose(p.data, expected)


def test_two_steps_different_gradients():
    p = Parameter(np.array([1.0]))

    optimizer = RMSProp([p], lr=0.1, alpha=0.9)

    p.grad = np.array([2.0])
    optimizer.step()

    avg = 0.1 * 4
    expected = 1.0 - 0.1 * 2 / np.sqrt(avg)

    np.testing.assert_allclose(p.data, [expected])

    p.grad = np.array([4.0])
    optimizer.step()

    avg = 0.9 * avg + 0.1 * 16
    expected = expected - 0.1 * 4 / np.sqrt(avg)

    np.testing.assert_allclose(p.data, [expected])