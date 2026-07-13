import numpy as np

import mytorch as torch
import mytorch.nn as nn
import mytorch.optim as optim

def test_forward_network_shape():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    x = torch.Tensor(np.random.randn(5, 4))

    y = model(x)

    assert y.shape == (5, 2)


def test_backward_populates_all_gradients():
    np.random.seed(0)

    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )

    criterion = nn.MSELoss()

    x = torch.Tensor(np.random.randn(8, 3))
    target = torch.Tensor(np.random.randn(8, 2))

    output = model(x)
    loss = criterion(output, target)

    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.shape


def test_optimizer_step_changes_parameters():
    np.random.seed(0)

    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    x = torch.Tensor(np.random.randn(10, 2))
    target = torch.Tensor(np.random.randn(10, 1))

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
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    x = torch.Tensor(np.random.randn(5, 2))
    target = torch.Tensor(np.random.randn(5, 1))

    loss = criterion(model(x), target)
    loss.backward()

    for p in model.parameters():
        assert p.grad is not None

    optimizer.zero_grad()

    for p in model.parameters():
        assert p.grad is None or np.all(p.grad == 0)


def test_loss_decreases_after_training():
    np.random.seed(0)

    model = nn.Sequential(
        nn.Linear(1, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    x = torch.Tensor(np.linspace(-1, 1, 100).reshape(-1, 1))
    target = torch.Tensor(2 * x.data + 1)

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
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )

    x = torch.Tensor(np.random.randn(6, 3))

    y1 = model(x)
    y2 = model(x)

    np.testing.assert_allclose(y1.data, y2.data)