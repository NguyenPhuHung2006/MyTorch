import numpy as np

from mytorch.nn import Parameter
from mytorch.optim import SGD


def test_sgd_single_parameter():
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([0.5, -0.5])

    optimizer = SGD([p], lr=0.1)
    optimizer.step()

    expected = np.array([0.95, 2.05])
    np.testing.assert_allclose(p.data, expected)


def test_sgd_multiple_parameters():
    p1 = Parameter(np.array([1.0]))
    p2 = Parameter(np.array([2.0]))

    p1.grad = np.array([2.0])
    p2.grad = np.array([-3.0])

    optimizer = SGD([p1, p2], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(p1.data, [0.8])
    np.testing.assert_allclose(p2.data, [2.3])


def test_skip_none_gradient():
    p = Parameter(np.array([5.0]))
    p.grad = None

    optimizer = SGD([p], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(p.data, [5.0])


def test_zero_gradient():
    p = Parameter(np.array([5.0]))
    p.grad = np.array([0.0])

    optimizer = SGD([p], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(p.data, [5.0])


def test_momentum_first_step():
    p = Parameter(np.array([1.0]))
    p.grad = np.array([2.0])

    optimizer = SGD([p], lr=0.1, momentum=0.9)
    optimizer.step()

    np.testing.assert_allclose(p.data, [0.8])


def test_momentum_accumulates():
    p = Parameter(np.array([1.0]))
    optimizer = SGD([p], lr=0.1, momentum=0.9)

    p.grad = np.array([1.0])
    optimizer.step()

    np.testing.assert_allclose(p.data, [0.9])

    p.grad = np.array([1.0])
    optimizer.step()

    # velocity = 0.9 * 1 + 1 = 1.9
    expected = 0.9 - 0.1 * 1.9
    np.testing.assert_allclose(p.data, [expected])


def test_momentum_changes_direction():
    p = Parameter(np.array([0.0]))
    optimizer = SGD([p], lr=1.0, momentum=0.9)

    p.grad = np.array([1.0])
    optimizer.step()

    np.testing.assert_allclose(p.data, [-1.0])

    p.grad = np.array([-1.0])
    optimizer.step()

    # velocity = 0.9 * 1 - 1 = -0.1
    # update = -(-0.1) = +0.1
    np.testing.assert_allclose(p.data, [-0.9])


def test_momentum_zero_equals_sgd():
    p1 = Parameter(np.array([3.0]))
    p2 = Parameter(np.array([3.0]))

    p1.grad = np.array([2.0])
    p2.grad = np.array([2.0])

    sgd = SGD([p1], lr=0.1)
    momentum = SGD([p2], lr=0.1, momentum=0.0)

    sgd.step()
    momentum.step()

    np.testing.assert_allclose(p1.data, p2.data)


def test_multiple_steps_without_momentum():
    p = Parameter(np.array([1.0]))
    optimizer = SGD([p], lr=0.1)

    for _ in range(5):
        p.grad = np.array([1.0])
        optimizer.step()

    np.testing.assert_allclose(p.data, [0.5])


def test_vector_parameter():
    p = Parameter(np.array([1.0, 2.0, 3.0]))
    p.grad = np.array([1.0, 2.0, 3.0])

    optimizer = SGD([p], lr=0.1)
    optimizer.step()

    expected = np.array([0.9, 1.8, 2.7])
    np.testing.assert_allclose(p.data, expected)