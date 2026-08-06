import numpy as np
import pytest

from mytorch import Tensor
from mytorch.nn import PositionalEncoding


def test_positional_encoding_output_shape():
    pe = PositionalEncoding(
        d_model=16,
        max_seq_len=100,
    )

    x = Tensor(np.random.randn(4, 20, 16))

    y = pe(x)

    assert y.shape == x.shape


def test_positional_encoding_position_zero():
    d_model = 8

    pe = PositionalEncoding(
        d_model=d_model,
        max_seq_len=10,
    )

    # At position 0:
    # sin(0) = 0
    # cos(0) = 1
    expected = np.array([
        0.0, 1.0,
        0.0, 1.0,
        0.0, 1.0,
        0.0, 1.0,
    ])

    assert np.allclose(pe.pe[0], expected)


def test_positional_encoding_is_deterministic():
    pe = PositionalEncoding(
        d_model=16,
        max_seq_len=100,
    )

    x = Tensor(np.random.randn(2, 20, 16))

    y1 = pe(x)
    y2 = pe(x)

    assert np.allclose(y1.data, y2.data)


def test_different_positions_have_different_encodings():
    pe = PositionalEncoding(
        d_model=16,
        max_seq_len=100,
    )

    assert not np.allclose(
        pe.pe[0],
        pe.pe[1],
    )

    assert not np.allclose(
        pe.pe[1],
        pe.pe[2],
    )


def test_positional_encoding_values_are_bounded():
    pe = PositionalEncoding(
        d_model=32,
        max_seq_len=1000,
    )

    assert np.all(pe.pe >= -1.0)
    assert np.all(pe.pe <= 1.0)


def test_positional_encoding_uses_correct_formula():
    d_model = 8
    max_seq_len = 10

    pe = PositionalEncoding(
        d_model=d_model,
        max_seq_len=max_seq_len,
    )

    position = np.arange(max_seq_len)[:, None]

    div_term = np.exp(
        np.arange(0, d_model, 2)
        * (-np.log(10000.0) / d_model)
    )

    expected = np.zeros((max_seq_len, d_model))

    expected[:, 0::2] = np.sin(position * div_term)
    expected[:, 1::2] = np.cos(position * div_term)

    assert np.allclose(
        pe.pe,
        expected,
    )


def test_positional_encoding_adds_to_input():
    d_model = 8

    pe = PositionalEncoding(
        d_model=d_model,
        max_seq_len=10,
        dropout=0.0,
    )

    x_data = np.random.randn(2, 5, d_model)
    x = Tensor(x_data)

    y = pe(x)

    expected = x_data + pe.pe[:5]

    assert np.allclose(
        y.data,
        expected,
    )


def test_positional_encoding_works_with_different_sequence_lengths():
    d_model = 16

    pe = PositionalEncoding(
        d_model=d_model,
        max_seq_len=100,
    )

    for seq_len in [1, 5, 10, 50, 100]:
        x = Tensor(
            np.random.randn(2, seq_len, d_model)
        )

        y = pe(x)

        assert y.shape == (2, seq_len, d_model)


def test_positional_encoding_uses_only_required_positions():
    d_model = 8

    pe = PositionalEncoding(
        d_model=d_model,
        max_seq_len=100,
    )

    x = Tensor(np.zeros((1, 5, d_model)))

    y = pe(x)

    expected = pe.pe[:5]

    assert np.allclose(
        y.data[0],
        expected,
    )


def test_positional_encoding_max_sequence_length():
    d_model = 8
    max_seq_len = 20

    pe = PositionalEncoding(
        d_model=d_model,
        max_seq_len=max_seq_len,
    )

    x = Tensor(
        np.random.randn(2, max_seq_len, d_model)
    )

    y = pe(x)

    assert y.shape == (
        2,
        max_seq_len,
        d_model,
    )


def test_positional_encoding_rejects_longer_sequence():
    d_model = 8
    max_seq_len = 10

    pe = PositionalEncoding(
        d_model=d_model,
        max_seq_len=max_seq_len,
    )

    x = Tensor(
        np.random.randn(2, max_seq_len + 1, d_model)
    )

    with pytest.raises(ValueError):
        pe(x)


def test_positional_encoding_does_not_modify_input():
    d_model = 8

    pe = PositionalEncoding(
        d_model=d_model,
        max_seq_len=10,
        dropout=0.0,
    )

    x_data = np.random.randn(2, 5, d_model)
    original = x_data.copy()

    x = Tensor(x_data)

    _ = pe(x)

    assert np.allclose(
        x.data,
        original,
    )


def test_positional_encoding_is_not_trainable():
    pe = PositionalEncoding(
        d_model=16,
        max_seq_len=100,
    )

    # The sinusoidal PE should not contain
    # any trainable parameters.
    parameters = list(pe.parameters())

    assert len(parameters) == 0
    
def test_odd_d_model():
    d_model = [2 * i + 1 for i in range(10)]
    
    for d in d_model:
        pe = PositionalEncoding(
            d,
            100
        )