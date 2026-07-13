import numpy as np
import pytest

from mytorch.tensor import Tensor
from mytorch.nn.modules.loss import CrossEntropyLoss


def manual_cross_entropy(logits, target):
    logits = np.asarray(logits, dtype=np.float64)

    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)

    loss = -np.log(probs[np.arange(len(target)), target])
    return loss


def test_forward_mean():
    logits = Tensor([[2.0, 1.0, 0.1],
                     [0.5, 2.5, 0.3]])
    target = Tensor([0, 1])

    criterion = CrossEntropyLoss()

    loss = criterion(logits, target)

    expected = manual_cross_entropy(
        [[2.0, 1.0, 0.1],
         [0.5, 2.5, 0.3]],
        np.array([0, 1]),
    ).mean()

    np.testing.assert_allclose(loss.data, expected)


def test_forward_sum():
    logits = Tensor([[2.0, 1.0],
                     [1.0, 2.0]])
    target = Tensor([0, 1])

    criterion = CrossEntropyLoss(reduction="sum")

    loss = criterion(logits, target)

    expected = manual_cross_entropy(
        [[2.0, 1.0],
         [1.0, 2.0]],
        np.array([0, 1]),
    ).sum()

    np.testing.assert_allclose(loss.data, expected)


def test_forward_none():
    logits = Tensor([[2.0, 1.0],
                     [1.0, 2.0]])
    target = Tensor([0, 1])

    criterion = CrossEntropyLoss(reduction="none")

    loss = criterion(logits, target)

    expected = manual_cross_entropy(
        [[2.0, 1.0],
         [1.0, 2.0]],
        np.array([0, 1]),
    )

    np.testing.assert_allclose(loss.data, expected)


def test_single_sample():
    logits = Tensor([[1.0, 3.0, 2.0]])
    target = Tensor([1])

    criterion = CrossEntropyLoss()

    loss = criterion(logits, target)

    expected = manual_cross_entropy(
        [[1.0, 3.0, 2.0]],
        np.array([1]),
    ).mean()

    np.testing.assert_allclose(loss.data, expected)


def test_perfect_prediction_small_loss():
    logits = Tensor([[10.0, -10.0]])

    target = Tensor([0])

    criterion = CrossEntropyLoss()

    loss = criterion(logits, target)

    assert loss.data < 1e-6


def test_uniform_logits():
    logits = Tensor([[0.0, 0.0, 0.0]])
    target = Tensor([2])

    criterion = CrossEntropyLoss()

    loss = criterion(logits, target)

    expected = np.log(3)

    np.testing.assert_allclose(loss.data, expected)


def test_backward_shape():
    logits = Tensor([[1.0, 2.0, 3.0],
                     [3.0, 2.0, 1.0]], requires_grad=True)
    target = Tensor([2, 0])

    criterion = CrossEntropyLoss()

    loss = criterion(logits, target)
    loss.backward()

    assert logits.grad.shape == logits.shape


def test_backward_not_zero():
    logits = Tensor([[1.0, 2.0],
                     [2.0, 1.0]], requires_grad=True)
    target = Tensor([1, 0])

    criterion = CrossEntropyLoss()

    loss = criterion(logits, target)
    loss.backward()

    assert np.any(logits.grad != 0)


def test_invalid_reduction():
    with pytest.raises(ValueError):
        CrossEntropyLoss(reduction="invalid")
        
def test_target_dimension_error():
    logits = Tensor([[1.0, 2.0]])
    target = Tensor([[0]])

    criterion = CrossEntropyLoss()

    with pytest.raises(ValueError):
        criterion(logits, target)


def test_batch_size_mismatch():
    logits = Tensor([[1.0, 2.0],
                     [3.0, 4.0]])
    target = Tensor([0])

    criterion = CrossEntropyLoss()

    with pytest.raises(ValueError):
        criterion(logits, target)


def test_target_out_of_range():
    logits = Tensor([[1.0, 2.0]])
    target = Tensor([2])

    criterion = CrossEntropyLoss()

    with pytest.raises((IndexError, ValueError)):
        criterion(logits, target)