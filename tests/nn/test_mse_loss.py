import numpy as np

from mytorch.tensor import Tensor
from mytorch.nn.modules.loss import MSELoss


def test_mse_loss_forward_mean():
    pred = Tensor(np.array([1.0, 2.0, 3.0]))
    target = Tensor(np.array([2.0, 2.0, 4.0]))

    criterion = MSELoss()

    loss = criterion(pred, target)

    expected = ((pred.data - target.data) ** 2).mean()

    np.testing.assert_allclose(loss.data, expected, rtol=1e-6)


def test_mse_loss_forward_sum():
    pred = Tensor(np.array([1.0, 2.0, 3.0]))
    target = Tensor(np.array([2.0, 2.0, 4.0]))

    criterion = MSELoss(reduction="sum")

    loss = criterion(pred, target)

    expected = ((pred.data - target.data) ** 2).sum()

    np.testing.assert_allclose(loss.data, expected, rtol=1e-6)


def test_mse_loss_forward_none():
    pred = Tensor(np.array([1.0, 2.0, 3.0]))
    target = Tensor(np.array([2.0, 2.0, 4.0]))

    criterion = MSELoss(reduction="none")

    loss = criterion(pred, target)

    expected = (pred.data - target.data) ** 2

    np.testing.assert_allclose(loss.data, expected, rtol=1e-6)


def test_mse_loss_zero():
    pred = Tensor(np.array([1.0, 2.0, 3.0]))
    target = Tensor(np.array([1.0, 2.0, 3.0]))

    criterion = MSELoss()

    loss = criterion(pred, target)

    np.testing.assert_allclose(loss.data, 0.0)


def test_mse_loss_backward_mean():
    pred = Tensor(
        np.array([1.0, 2.0, 3.0]),
        requires_grad=True,
    )

    target = Tensor(np.array([2.0, 2.0, 4.0]))

    criterion = MSELoss()

    loss = criterion(pred, target)
    loss.backward()

    expected = 2 * (pred.data - target.data) / pred.data.size

    np.testing.assert_allclose(
        pred.grad,
        expected,
        rtol=1e-6,
    )


def test_mse_loss_backward_sum():
    pred = Tensor(
        np.array([1.0, 2.0, 3.0]),
        requires_grad=True,
    )

    target = Tensor(np.array([2.0, 2.0, 4.0]))

    criterion = MSELoss(reduction="sum")

    loss = criterion(pred, target)
    loss.backward()

    expected = 2 * (pred.data - target.data)

    np.testing.assert_allclose(
        pred.grad,
        expected,
        rtol=1e-6,
    )


def test_mse_loss_no_parameters():
    criterion = MSELoss()

    assert list(criterion.parameters()) == []


def test_mse_loss_train_eval():
    criterion = MSELoss()

    criterion.eval()
    assert criterion.training is False

    criterion.train()
    assert criterion.training is True