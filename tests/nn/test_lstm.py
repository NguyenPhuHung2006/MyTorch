import numpy as np

import mytorch as torch
from mytorch import Tensor
from mytorch import nn


# ============================================================
# Helpers
# ============================================================

def make_input(seq_len=5, batch_size=3, input_size=4):
    return Tensor(
        np.random.randn(seq_len, batch_size, input_size)
    )


# ============================================================
# LSTMCell
# ============================================================

def test_lstm_cell_output_shape():
    cell = nn.LSTMCell(
        input_size=4,
        hidden_size=6,
    )

    x = Tensor(np.random.randn(3, 4))
    h = Tensor(np.random.randn(3, 6))
    c = Tensor(np.random.randn(3, 6))

    h_new, c_new = cell(x, h, c)

    assert h_new.shape == (3, 6)
    assert c_new.shape == (3, 6)


def test_lstm_cell_default_state():
    cell = nn.LSTMCell(
        input_size=4,
        hidden_size=6,
    )

    x = Tensor(np.random.randn(3, 4))

    h, c = cell(x)

    assert h.shape == (3, 6)
    assert c.shape == (3, 6)


def test_lstm_cell_zero_input():
    cell = nn.LSTMCell(
        input_size=4,
        hidden_size=6,
        bias=False,
    )

    x = Tensor(np.zeros((3, 4)))
    h = Tensor(np.zeros((3, 6)))
    c = Tensor(np.zeros((3, 6)))

    h_new, c_new = cell(x, h, c)

    assert np.allclose(h_new.numpy(), 0)
    assert np.allclose(c_new.numpy(), 0)


def test_lstm_cell_invalid_input_dimension():
    cell = nn.LSTMCell(
        input_size=4,
        hidden_size=6,
    )

    x = Tensor(np.random.randn(3, 2, 4))

    try:
        cell(x)
        assert False
    except ValueError:
        pass


def test_lstm_cell_invalid_input_size():
    cell = nn.LSTMCell(
        input_size=4,
        hidden_size=6,
    )

    x = Tensor(np.random.randn(3, 5))

    try:
        cell(x)
        assert False
    except ValueError:
        pass


def test_lstm_cell_invalid_hidden_shape():
    cell = nn.LSTMCell(
        input_size=4,
        hidden_size=6,
    )

    x = Tensor(np.random.randn(3, 4))
    h = Tensor(np.random.randn(3, 5))
    c = Tensor(np.random.randn(3, 6))

    try:
        cell(x, h, c)
        assert False
    except ValueError:
        pass


def test_lstm_cell_invalid_cell_shape():
    cell = nn.LSTMCell(
        input_size=4,
        hidden_size=6,
    )

    x = Tensor(np.random.randn(3, 4))
    h = Tensor(np.random.randn(3, 6))
    c = Tensor(np.random.randn(3, 5))

    try:
        cell(x, h, c)
        assert False
    except ValueError:
        pass


# ============================================================
# Basic LSTM
# ============================================================

def test_lstm_output_shape():
    lstm = nn.LSTM(
        input_size=4,
        hidden_size=6,
    )

    x = make_input()

    outputs, hiddens, cells = lstm(x)

    assert outputs.shape == (5, 3, 6)
    assert hiddens.shape == (1, 3, 6)
    assert cells.shape == (1, 3, 6)


def test_lstm_batch_first():
    lstm = nn.LSTM(
        input_size=4,
        hidden_size=6,
        batch_first=True,
    )

    x = Tensor(
        np.random.randn(3, 5, 4)
    )

    outputs, hiddens, cells = lstm(x)

    assert outputs.shape == (3, 5, 6)
    assert hiddens.shape == (1, 3, 6)
    assert cells.shape == (1, 3, 6)


def test_lstm_zero_input():
    lstm = nn.LSTM(
        input_size=4,
        hidden_size=6,
        bias=False,
    )

    x = Tensor(
        np.zeros((5, 2, 4))
    )

    outputs, hiddens, cells = lstm(x)

    assert np.allclose(outputs.numpy(), 0)
    assert np.allclose(hiddens.numpy(), 0)
    assert np.allclose(cells.numpy(), 0)


# ============================================================
# Initial state
# ============================================================

def test_lstm_initial_state():
    lstm = nn.LSTM(
        input_size=4,
        hidden_size=6,
    )

    x = Tensor(
        np.random.randn(5, 2, 4)
    )

    h0 = Tensor(
        np.ones((1, 2, 6))
    )

    c0 = Tensor(
        np.ones((1, 2, 6))
    )

    outputs, hiddens, cells = lstm(
        x,
        h0,
        c0,
    )

    assert outputs.shape == (5, 2, 6)
    assert hiddens.shape == (1, 2, 6)
    assert cells.shape == (1, 2, 6)


# ============================================================
# Recurrence
# ============================================================

def test_lstm_recurrence():
    lstm = nn.LSTM(
        input_size=2,
        hidden_size=3,
        bias=False,
    )

    x = Tensor(
        np.random.randn(4, 1, 2)
    )

    h0 = Tensor(
        np.random.randn(1, 1, 3)
    )

    c0 = Tensor(
        np.random.randn(1, 1, 3)
    )

    outputs, hiddens, cells = lstm(
        x,
        h0,
        c0,
    )

    h = h0[0]
    c = c0[0]

    expected_outputs = []

    for t in range(4):
        h, c = lstm.cells[0][0](x[t], h, c)
        expected_outputs.append(h)

    expected = torch.stack(
        expected_outputs,
        axis=0,
    )

    assert np.allclose(
        outputs.numpy(),
        expected.numpy(),
    )

    assert np.allclose(
        hiddens[0].numpy(),
        h.numpy(),
    )

    assert np.allclose(
        cells[0].numpy(),
        c.numpy(),
    )


# ============================================================
# Returned states
# ============================================================

def test_lstm_returns_outputs_hiddens_cells():
    lstm = nn.LSTM(
        input_size=4,
        hidden_size=6,
        num_layers=3,
    )

    x = Tensor(
        np.random.randn(5, 2, 4)
    )

    outputs, hiddens, cells = lstm(x)

    assert outputs.shape == (5, 2, 6)
    assert hiddens.shape == (3, 2, 6)
    assert cells.shape == (3, 2, 6)


def test_lstm_returned_final_states():
    lstm = nn.LSTM(
        input_size=3,
        hidden_size=4,
        num_layers=2,
    )

    x = Tensor(
        np.random.randn(5, 2, 3)
    )

    outputs, hiddens, cells = lstm(x)

    assert np.allclose(
        outputs[-1].numpy(),
        hiddens[-1].numpy(),
    )


# ============================================================
# Multiple layers
# ============================================================

def test_lstm_num_layers():
    lstm = nn.LSTM(
        input_size=4,
        hidden_size=6,
        num_layers=3,
    )

    x = make_input()

    outputs, hiddens, cells = lstm(x)

    assert outputs.shape == (5, 3, 6)
    assert hiddens.shape == (3, 3, 6)
    assert cells.shape == (3, 3, 6)


# ============================================================
# Zero sequence length
# ============================================================

def test_lstm_zero_sequence_length():
    lstm = nn.LSTM(
        input_size=4,
        hidden_size=6,
    )

    x = Tensor(
        np.empty((0, 2, 4))
    )

    outputs, hiddens, cells = lstm(x)

    assert outputs.shape == (0, 2, 6)
    assert hiddens.shape == (1, 2, 6)
    assert cells.shape == (1, 2, 6)


# ============================================================
# Backward
# ============================================================

def test_lstm_backward():
    lstm = nn.LSTM(
        input_size=2,
        hidden_size=2,
    )

    x = Tensor(
        np.random.randn(3, 1, 2),
        requires_grad=True,
    )

    outputs, hiddens, cells = lstm(x)

    loss = outputs.sum()

    loss.backward()

    assert x.grad is not None

    for name, param in lstm.named_parameters():
        assert param.grad is not None, (
            f"{name} has no gradient"
        )


def test_lstm_backward_multiple_layers():
    lstm = nn.LSTM(
        input_size=3,
        hidden_size=4,
        num_layers=2,
    )

    x = Tensor(
        np.random.randn(4, 2, 3),
        requires_grad=True,
    )

    outputs, hiddens, cells = lstm(x)

    loss = outputs.sum()

    loss.backward()

    assert x.grad is not None

    for name, param in lstm.named_parameters():
        assert param.grad is not None, (
            f"{name} has no gradient"
        )