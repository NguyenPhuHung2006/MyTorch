import numpy as np

from mytorch.tensor import Tensor
from mytorch.nn.modules.loss import BCELoss

def test_bce_loss_forward_mean():
    pred = Tensor(np.array([0.9, 0.2, 0.8]))
    target = Tensor(np.array([1.0, 0.0, 1.0]))

    criterion = BCELoss()

    loss = criterion(pred, target)

    expected = -(
        target.data * np.log(pred.data)
        + (1 - target.data) * np.log(1 - pred.data)
    ).mean()

    np.testing.assert_allclose(loss.data, expected, rtol=1e-6)


def test_bce_loss_forward_sum():
    pred = Tensor(np.array([0.9, 0.2, 0.8]))
    target = Tensor(np.array([1.0, 0.0, 1.0]))

    criterion = BCELoss(reduction="sum")

    loss = criterion(pred, target)

    expected = -(
        target.data * np.log(pred.data)
        + (1 - target.data) * np.log(1 - pred.data)
    ).sum()

    np.testing.assert_allclose(loss.data, expected, rtol=1e-6)


def test_bce_loss_forward_none():
    pred = Tensor(np.array([0.9, 0.2, 0.8]))
    target = Tensor(np.array([1.0, 0.0, 1.0]))

    criterion = BCELoss(reduction="none")

    loss = criterion(pred, target)

    expected = -(
        target.data * np.log(pred.data)
        + (1 - target.data) * np.log(1 - pred.data)
    )

    np.testing.assert_allclose(loss.data, expected, rtol=1e-6)


def test_bce_loss_zero():
    pred = Tensor(np.array([0.999999, 0.000001]))
    target = Tensor(np.array([1.0, 0.0]))

    criterion = BCELoss()

    loss = criterion(pred, target)

    assert loss.data < 1e-5


def test_bce_loss_backward_mean():
    pred = Tensor(
        np.array([0.9, 0.2, 0.8]),
        requires_grad=True,
    )
    target = Tensor(np.array([1.0, 0.0, 1.0]))

    criterion = BCELoss()

    loss = criterion(pred, target)
    loss.backward()

    n = pred.data.size
    expected = (
        (pred.data - target.data)
        / (pred.data * (1 - pred.data) * n)
    )

    np.testing.assert_allclose(
        pred.grad,
        expected,
        rtol=1e-6,
    )


def test_bce_loss_backward_sum():
    pred = Tensor(
        np.array([0.9, 0.2, 0.8]),
        requires_grad=True,
    )
    target = Tensor(np.array([1.0, 0.0, 1.0]))

    criterion = BCELoss(reduction="sum")

    loss = criterion(pred, target)
    loss.backward()

    expected = (
        (pred.data - target.data)
        / (pred.data * (1 - pred.data))
    )

    np.testing.assert_allclose(
        pred.grad,
        expected,
        rtol=1e-6,
    )


def test_bce_loss_no_parameters():
    criterion = BCELoss()

    assert list(criterion.parameters()) == []


def test_bce_loss_train_eval():
    criterion = BCELoss()

    criterion.eval()
    assert criterion.training is False

    criterion.train()
    assert criterion.training is True
    
def test_bce_loss_gradient_numerical():
    eps = 1e-6

    x_data = np.random.uniform(0.1, 0.9, size=5)
    target = np.random.randint(0, 2, size=5).astype(float)

    pred = Tensor(x_data.copy(), requires_grad=True)
    criterion = BCELoss()

    loss = criterion(pred, Tensor(target))
    loss.backward()

    numerical = np.zeros_like(x_data)

    for i in range(len(x_data)):
        plus = x_data.copy()
        minus = x_data.copy()

        plus[i] += eps
        minus[i] -= eps

        f1 = -(
            target * np.log(plus)
            + (1 - target) * np.log(1 - plus)
        ).mean()

        f2 = -(
            target * np.log(minus)
            + (1 - target) * np.log(1 - minus)
        ).mean()

        numerical[i] = (f1 - f2) / (2 * eps)

    np.testing.assert_allclose(
        pred.grad,
        numerical,
        rtol=1e-4,
        atol=1e-6,
    )