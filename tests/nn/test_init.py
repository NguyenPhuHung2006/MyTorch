import math
import numpy as np
import pytest

from mytorch import Tensor
from mytorch.nn import init


# ==========================================================
# uniform_
# ==========================================================

def test_uniform_range():
    x = Tensor(np.empty((10000,)))

    init.uniform_(x, -2, 3)

    assert np.all(x.data >= -2)
    assert np.all(x.data <= 3)


def test_uniform_mean():
    x = Tensor(np.empty((100000,)))

    init.uniform_(x, -1, 1)

    assert abs(x.data.mean()) < 0.02


def test_uniform_variance():
    x = Tensor(np.empty((100000,)))

    init.uniform_(x, -1, 1)

    expected = (2 ** 2) / 12  # (b - a)^2 / 12

    assert np.isclose(x.data.var(), expected, atol=0.02)


def test_uniform_returns_same_tensor():
    x = Tensor(np.empty((10, 10)))

    y = init.uniform_(x)

    assert y is x


# ==========================================================
# normal_
# ==========================================================

def test_normal_mean():
    x = Tensor(np.empty((100000,)))

    init.normal_(x, mean=5, std=2)

    assert np.isclose(x.data.mean(), 5, atol=0.05)


def test_normal_std():
    x = Tensor(np.empty((100000,)))

    init.normal_(x, mean=0, std=3)

    assert np.isclose(x.data.std(), 3, atol=0.05)


def test_normal_returns_same_tensor():
    x = Tensor(np.empty((10, 10)))

    y = init.normal_(x)

    assert y is x


# ==========================================================
# calculate_gain
# ==========================================================

def test_gain_linear():
    assert init.calculate_gain("linear") == 1.0


def test_gain_sigmoid():
    assert init.calculate_gain("sigmoid") == 1.0


def test_gain_relu():
    assert init.calculate_gain("relu") == math.sqrt(2)


def test_gain_tanh():
    assert init.calculate_gain("tanh") == 5 / 3


def test_gain_leaky_relu():
    expected = math.sqrt(2 / (1 + 0.2 ** 2))
    assert init.calculate_gain("leaky_relu", 0.2) == expected


# ==========================================================
# fan_in / fan_out
# ==========================================================

def test_calculate_fan():
    w = Tensor(np.empty((32, 64)))

    fan_in, fan_out = init._calculate_fan_in_and_fan_out(w)

    assert fan_in == 64
    assert fan_out == 32


# ==========================================================
# xavier_uniform_
# ==========================================================

def test_xavier_uniform_variance():
    w = Tensor(np.empty((64, 128)))

    init.xavier_uniform_(w)

    expected = 2 / (64 + 128)

    assert np.isclose(w.data.var(), expected, rtol=0.15)


def test_xavier_uniform_bounds():
    w = Tensor(np.empty((64, 128)))

    init.xavier_uniform_(w)

    bound = math.sqrt(6 / (64 + 128))

    assert np.all(w.data >= -bound)
    assert np.all(w.data <= bound)


# ==========================================================
# xavier_normal_
# ==========================================================

def test_xavier_normal_variance():
    w = Tensor(np.empty((64, 128)))

    init.xavier_normal_(w)

    expected = 2 / (64 + 128)

    assert np.isclose(w.data.var(), expected, rtol=0.15)


# ==========================================================
# kaiming_uniform_
# ==========================================================

def test_kaiming_uniform_variance():
    w = Tensor(np.empty((64, 128)))

    init.kaiming_uniform_(w, nonlinearity="relu")

    expected = 2 / 128

    assert np.isclose(w.data.var(), expected, rtol=0.15)


def test_kaiming_uniform_bounds():
    w = Tensor(np.empty((64, 128)))

    init.kaiming_uniform_(w, nonlinearity="relu")

    bound = math.sqrt(6 / 128)

    assert np.all(w.data >= -bound)
    assert np.all(w.data <= bound)


# ==========================================================
# kaiming_normal_
# ==========================================================

def test_kaiming_normal_variance():
    w = Tensor(np.empty((64, 128)))

    init.kaiming_normal_(w, nonlinearity="relu")

    expected = 2 / 128

    assert np.isclose(w.data.var(), expected, rtol=0.15)


# ==========================================================
# error handling
# ==========================================================

def test_invalid_nonlinearity():
    with pytest.raises(ValueError):
        init.calculate_gain("unknown")


def test_invalid_fan():
    x = Tensor(np.empty((10,)))

    with pytest.raises(ValueError):
        init._calculate_fan_in_and_fan_out(x)


def test_invalid_mode():
    x = Tensor(np.empty((10, 10)))

    with pytest.raises(ValueError):
        init.kaiming_uniform_(x, mode="invalid")