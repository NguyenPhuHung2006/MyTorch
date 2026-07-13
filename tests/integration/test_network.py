import numpy as np

from mytorch.tensor import Tensor
from mytorch.nn.modules.linear import Linear
from mytorch.nn.modules.sequential import Sequential
from mytorch.nn.modules.activations import ReLU
from mytorch.nn.modules.loss import MSELoss
from mytorch.optim.sgd import SGD

def test_forward_network_shape():
    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2),
    )

    x = Tensor(np.random.randn(5, 4))

    y = model(x)

    assert y.shape == (5, 2)


def test_backward_populates_all_gradients():
    np.random.seed(0)

    model = Sequential(
        Linear(3, 5),
        ReLU(),
        Linear(5, 2),
    )

    criterion = MSELoss()

    x = Tensor(np.random.randn(8, 3))
    target = Tensor(np.random.randn(8, 2))

    output = model(x)
    loss = criterion(output, target)

    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.shape


def test_optimizer_step_changes_parameters():
    np.random.seed(0)

    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
    )

    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    x = Tensor(np.random.randn(10, 2))
    target = Tensor(np.random.randn(10, 1))

    before = [p.data.copy() for p in model.parameters()]

    loss = criterion(model(x), target)
    loss.backward()
    optimizer.step()

    after = [p.data for p in model.parameters()]

    assert any(
        not np.array_equal(a, b)
        for a, b in zip(before, after)
    )


def test_zero_grad_clears_gradients():
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 1),
    )

    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    x = Tensor(np.random.randn(5, 2))
    target = Tensor(np.random.randn(5, 1))

    loss = criterion(model(x), target)
    loss.backward()

    for p in model.parameters():
        assert p.grad is not None

    optimizer.zero_grad()

    for p in model.parameters():
        assert p.grad is None or np.all(p.grad == 0)


def test_loss_decreases_after_training():
    np.random.seed(0)

    model = Sequential(
        Linear(1, 8),
        ReLU(),
        Linear(8, 1),
    )

    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.05)

    x = Tensor(np.linspace(-1, 1, 100).reshape(-1, 1))
    target = Tensor(2 * x.data + 1)

    initial_loss = criterion(model(x), target).data

    for _ in range(200):
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), target).data

    assert final_loss < initial_loss


def test_network_multiple_forward_calls():
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2),
    )

    x = Tensor(np.random.randn(6, 3))

    y1 = model(x)
    y2 = model(x)

    np.testing.assert_allclose(y1.data, y2.data)