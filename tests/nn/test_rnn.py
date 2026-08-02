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
    
import pytest

import mytorch as torch

# ============================================================
# Helpers
# ============================================================

def make_input(seq_len=4, batch_size=3, input_size=4):
    np.random.seed(42)
    return torch.Tensor(
        np.random.randn(seq_len, batch_size, input_size)
    )


# ============================================================
# Basic shapes
# ============================================================

def test_rnn_output_shape():
    rnn = nn.RNN(4, 6)

    x = make_input()
    output, h_n = rnn(x)

    assert output.shape == (4, 3, 6)
    assert h_n.shape == (1, 3, 6)


def test_rnn_multiple_layers():
    rnn = nn.RNN(4, 6, num_layers=3)

    output, h_n = rnn(make_input())

    assert output.shape == (4, 3, 6)
    assert h_n.shape == (3, 3, 6)


def test_rnn_bidirectional():
    rnn = nn.RNN(4, 6, bidirectional=True)

    output, h_n = rnn(make_input())

    assert output.shape == (4, 3, 12)
    assert h_n.shape == (2, 3, 6)


def test_rnn_bidirectional_multiple_layers():
    rnn = nn.RNN(
        4,
        6,
        num_layers=2,
        bidirectional=True,
    )

    output, h_n = rnn(make_input())

    assert output.shape == (4, 3, 12)
    assert h_n.shape == (4, 3, 6)


# ============================================================
# batch_first
# ============================================================

def test_rnn_batch_first():
    rnn = nn.RNN(4, 6, batch_first=True)

    x = torch.Tensor(
        np.random.randn(3, 4, 4)
    )

    output, h_n = rnn(x)

    assert output.shape == (3, 4, 6)
    assert h_n.shape == (1, 3, 6)


# ============================================================
# Hidden state
# ============================================================

def test_rnn_custom_hidden_state():
    rnn = nn.RNN(4, 6)

    x = make_input()

    h0 = torch.Tensor(
        np.zeros((1, 3, 6))
    )

    output, h_n = rnn(x, h0)

    assert output.shape == (4, 3, 6)
    assert h_n.shape == (1, 3, 6)


def test_rnn_hidden_state_affects_output():
    rnn = nn.RNN(4, 6)

    x = torch.Tensor(
        np.zeros((3, 2, 4))
    )

    h_zero = torch.Tensor(
        np.zeros((1, 2, 6))
    )

    h_one = torch.Tensor(
        np.ones((1, 2, 6))
    )

    out_zero, _ = rnn(x, h_zero)
    out_one, _ = rnn(x, h_one)

    assert not np.allclose(
        out_zero.numpy(),
        out_one.numpy(),
    )


# ============================================================
# Activations
# ============================================================

def test_rnn_tanh():
    rnn = nn.RNN(
        4,
        6,
        nonlinearity="tanh",
    )

    output, _ = rnn(make_input())

    assert np.all(output.numpy() <= 1)
    assert np.all(output.numpy() >= -1)


def test_rnn_relu():
    rnn = nn.RNN(
        4,
        6,
        nonlinearity="relu",
    )

    output, _ = rnn(make_input())

    assert np.all(output.numpy() >= 0)


def test_rnn_invalid_nonlinearity():
    with pytest.raises(ValueError):
        nn.RNN(
            4,
            6,
            nonlinearity="invalid",
        )


# ============================================================
# Zero-length sequence
# ============================================================

def test_rnn_zero_sequence():
    rnn = nn.RNN(4, 6)

    x = torch.Tensor(
        np.empty((0, 3, 4))
    )

    output, h_n = rnn(x)

    assert output.shape == (0, 3, 6)
    assert h_n.shape == (1, 3, 6)


def test_rnn_zero_sequence_bidirectional():
    rnn = nn.RNN(
        4,
        6,
        bidirectional=True,
    )

    x = torch.Tensor(
        np.empty((0, 3, 4))
    )

    output, h_n = rnn(x)

    assert output.shape == (0, 3, 12)
    assert h_n.shape == (2, 3, 6)


# ============================================================
# Output / hidden-state relationship
# ============================================================

def test_rnn_final_hidden_matches_last_output():
    rnn = nn.RNN(4, 6)

    output, h_n = rnn(make_input())

    np.testing.assert_allclose(
        output[-1].numpy(),
        h_n[0].numpy(),
    )


def test_rnn_bidirectional_final_hidden():
    rnn = nn.RNN(
        4,
        6,
        bidirectional=True,
    )

    output, h_n = rnn(make_input())

    # Forward final hidden
    np.testing.assert_allclose(
        output[-1, :, :6].numpy(),
        h_n[0].numpy(),
    )

    # Backward final hidden
    np.testing.assert_allclose(
        output[0, :, 6:].numpy(),
        h_n[1].numpy(),
    )


# ============================================================
# Input validation
# ============================================================

def test_rnn_rejects_wrong_input_dimensions():
    rnn = nn.RNN(4, 6)

    x = torch.Tensor(
        np.random.randn(3, 4)
    )

    with pytest.raises(ValueError):
        rnn(x)


def test_rnn_rejects_wrong_input_size():
    rnn = nn.RNN(4, 6)

    x = torch.Tensor(
        np.random.randn(3, 2, 5)
    )

    with pytest.raises(ValueError):
        rnn(x)


def test_rnn_rejects_wrong_hidden_shape():
    rnn = nn.RNN(4, 6)

    x = make_input()

    h = torch.Tensor(
        np.zeros((2, 3, 6))
    )

    with pytest.raises(ValueError):
        rnn(x, h)


# ============================================================
# Bias
# ============================================================

def test_rnn_without_bias():
    rnn = nn.RNN(
        4,
        6,
        bias=False,
    )

    # Just make sure forward works.
    output, h_n = rnn(make_input())

    assert output.shape == (4, 3, 6)
    assert h_n.shape == (1, 3, 6)


# ============================================================
# Dropout
# ============================================================

def test_rnn_dropout():
    rnn = nn.RNN(
        4,
        6,
        num_layers=2,
        dropout=0.5,
    )

    output, h_n = rnn(make_input())

    assert output.shape == (4, 3, 6)
    assert h_n.shape == (2, 3, 6)


# ============================================================
# Autograd
# ============================================================

def test_rnn_backward():
    rnn = nn.RNN(4, 6)

    x = torch.Tensor(
        np.random.randn(3, 2, 4),
        requires_grad=True,
    )

    output, _ = rnn(x)

    loss = output.sum()
    loss.backward()

    assert x.grad is not None

    for p in rnn.parameters():
        assert p.grad is not None
