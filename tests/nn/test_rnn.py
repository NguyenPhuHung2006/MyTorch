import numpy as np

from mytorch import Tensor
from mytorch import nn


def test_rnn_output_shape():
    rnn = nn.RNN(
        input_size=10,
        hidden_size=20,
        num_layers=2,
    )

    x = Tensor(np.random.randn(5, 4, 10))

    output, h = rnn(x)

    assert output.shape == (5, 4, 20)
    assert h.shape == (2, 4, 20)


def test_rnn_batch_first():
    rnn = nn.RNN(
        input_size=10,
        hidden_size=20,
        num_layers=2,
        batch_first=True,
    )

    x = Tensor(np.random.randn(4, 5, 10))

    output, h = rnn(x)

    assert output.shape == (4, 5, 20)
    assert h.shape == (2, 4, 20)


def test_rnn_single_layer():
    rnn = nn.RNN(
        input_size=8,
        hidden_size=16,
        num_layers=1,
    )

    x = Tensor(np.random.randn(7, 3, 8))

    output, h = rnn(x)

    assert output.shape == (7, 3, 16)
    assert h.shape == (1, 3, 16)


def test_rnn_custom_hidden_state():
    rnn = nn.RNN(
        input_size=6,
        hidden_size=9,
        num_layers=3,
    )

    x = Tensor(np.random.randn(4, 2, 6))
    hx = Tensor(np.random.randn(3, 2, 9))

    output, h = rnn(x, hx)

    assert output.shape == (4, 2, 9)
    assert h.shape == (3, 2, 9)


def test_rnn_backward():
    rnn = nn.RNN(
        input_size=5,
        hidden_size=7,
        num_layers=2,
    )

    x = Tensor(
        np.random.randn(6, 4, 5),
        requires_grad=True,
    )

    output, h = rnn(x)

    loss = output.sum() + h.sum()
    loss.backward()

    assert x.grad is not None

    for p in rnn.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.shape


def test_rnn_zero_sequence_length():
    rnn = nn.RNN(
        input_size=5,
        hidden_size=7,
    )

    x = Tensor(np.random.randn(0, 4, 5))

    output, h = rnn(x)

    assert output.shape == (0, 4, 7)
    assert h.shape == (1, 4, 7)


def test_rnn_zero_input():
    rnn = nn.RNN(
        input_size=4,
        hidden_size=6,
    )

    x = Tensor(np.zeros((5, 3, 4)))

    output, h = rnn(x)

    assert output.shape == (5, 3, 6)
    assert h.shape == (1, 3, 6)


def test_rnn_parameters_count():
    rnn = nn.RNN(
        input_size=4,
        hidden_size=5,
        num_layers=2,
    )

    params = list(rnn.parameters())

    assert len(params) == 6


def test_rnn_forward_is_deterministic():
    np.random.seed(0)

    rnn = nn.RNN(
        input_size=5,
        hidden_size=4,
    )

    x = Tensor(np.random.randn(3, 2, 5))

    y1, h1 = rnn(x)
    y2, h2 = rnn(x)

    np.testing.assert_allclose(y1.numpy(), y2.numpy())
    np.testing.assert_allclose(h1.numpy(), h2.numpy())