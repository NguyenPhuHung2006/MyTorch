import numpy as np
import pytest

import mytorch as torch
from mytorch import nn


# ============================================================
# Helpers
# ============================================================

def assert_allclose(actual, expected, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(
        actual.numpy() if hasattr(actual, "numpy") else actual,
        expected,
        atol=atol,
        rtol=rtol,
    )


def make_input(shape=(8, 10, 16)):
    np.random.seed(42)
    return torch.Tensor(np.random.randn(*shape))


# ============================================================
# Forward / Shape
# ============================================================

def test_layernorm_output_shape():
    x = make_input((8, 10, 16))

    ln = nn.LayerNorm(16)

    y = ln(x)

    assert y.shape == x.shape


def test_layernorm_2d_input():
    x = make_input((8, 16))

    ln = nn.LayerNorm(16)

    y = ln(x)

    assert y.shape == (8, 16)


def test_layernorm_3d_input():
    # Typical Transformer input:
    # (batch, sequence_length, embedding_dim)
    x = make_input((8, 10, 16))

    ln = nn.LayerNorm(16)

    y = ln(x)

    assert y.shape == (8, 10, 16)


# ============================================================
# Normalization
# ============================================================

def test_layernorm_zero_mean():
    x = make_input((8, 10, 16))

    ln = nn.LayerNorm(16)

    y = ln(x)

    y_np = y.numpy()

    # LayerNorm normalizes the last dimension.
    mean = y_np.mean(axis=-1)

    np.testing.assert_allclose(
        mean,
        0.0,
        atol=1e-5,
    )


def test_layernorm_unit_variance():
    x = make_input((8, 10, 16))

    ln = nn.LayerNorm(16)

    y = ln(x)

    y_np = y.numpy()

    # Variance is computed over the normalized dimension.
    var = y_np.var(axis=-1)

    np.testing.assert_allclose(
        var,
        1.0,
        atol=1e-4,
    )


def test_layernorm_each_token_independently():
    x = make_input((4, 6, 8))

    ln = nn.LayerNorm(8)

    y = ln(x)

    y_np = y.numpy()

    means = y_np.mean(axis=-1)
    variances = y_np.var(axis=-1)

    assert means.shape == (4, 6)
    assert variances.shape == (4, 6)

    np.testing.assert_allclose(
        means,
        0.0,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        variances,
        1.0,
        atol=1e-4,
    )


# ============================================================
# Reference implementation
# ============================================================

def test_layernorm_matches_reference():
    x_np = np.random.randn(4, 6, 8)

    x = torch.Tensor(x_np)

    ln = nn.LayerNorm(8)

    y = ln(x)

    eps = ln.eps

    mean = x_np.mean(axis=-1, keepdims=True)
    var = x_np.var(axis=-1, keepdims=True)

    expected = (x_np - mean) / np.sqrt(var + eps)

    # Default weight = 1 and bias = 0
    np.testing.assert_allclose(
        y.numpy(),
        expected,
        atol=1e-5,
        rtol=1e-5,
    )


# ============================================================
# Affine parameters
# ============================================================

def test_layernorm_has_weight_and_bias():
    ln = nn.LayerNorm(16)

    assert ln.weight is not None
    assert ln.bias is not None

    assert ln.weight.shape == (16,)
    assert ln.bias.shape == (16,)


def test_layernorm_default_weight():
    ln = nn.LayerNorm(16)

    np.testing.assert_allclose(
        ln.weight.numpy(),
        np.ones(16),
    )


def test_layernorm_default_bias():
    ln = nn.LayerNorm(16)

    np.testing.assert_allclose(
        ln.bias.numpy(),
        np.zeros(16),
    )


def test_layernorm_affine_parameters_change_output():
    x = make_input((4, 8))

    ln = nn.LayerNorm(8)

    # Change gamma and beta
    ln.weight.data[...] = 2.0
    ln.bias.data[...] = 3.0

    y = ln(x)

    y_np = y.numpy()

    # y = gamma * normalized_x + beta
    mean = y_np.mean(axis=-1)

    np.testing.assert_allclose(
        mean,
        3.0,
        atol=1e-5,
    )

    variance = y_np.var(axis=-1)

    np.testing.assert_allclose(
        variance,
        4.0,
        atol=1e-4,
    )


# ============================================================
# elementwise_affine=False
# ============================================================

def test_layernorm_without_affine():
    x = make_input((4, 8))

    ln = nn.LayerNorm(
        8,
        elementwise_affine=False,
    )

    y = ln(x)

    y_np = y.numpy()

    mean = y_np.mean(axis=-1)
    variance = y_np.var(axis=-1)

    np.testing.assert_allclose(
        mean,
        0.0,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        variance,
        1.0,
        atol=1e-4,
    )


def test_layernorm_without_affine_has_no_parameters():
    ln = nn.LayerNorm(
        16,
        elementwise_affine=False,
    )

    assert ln.weight is None
    assert ln.bias is None


# ============================================================
# Constant input
# ============================================================

def test_layernorm_constant_input():
    x = torch.Tensor(
        np.ones((4, 8)) * 5.0
    )

    ln = nn.LayerNorm(8)

    y = ln(x)

    # (x - mean) = 0
    # Therefore normalized output should be 0.
    np.testing.assert_allclose(
        y.numpy(),
        np.zeros((4, 8)),
        atol=1e-5,
    )


# ============================================================
# Numerical stability
# ============================================================

def test_layernorm_near_constant_input():
    x_np = np.ones((4, 8)) * 5.0
    x_np[:, 0] += 1e-7

    x = torch.Tensor(x_np)

    ln = nn.LayerNorm(8)

    y = ln(x)

    assert np.all(np.isfinite(y.numpy()))


def test_layernorm_large_values():
    x_np = np.random.randn(4, 8) * 1e6

    x = torch.Tensor(x_np)

    ln = nn.LayerNorm(8)

    y = ln(x)

    assert np.all(np.isfinite(y.numpy()))

    np.testing.assert_allclose(
        y.numpy().mean(axis=-1),
        0.0,
        atol=1e-4,
    )


# ============================================================
# Different normalized shapes
# ============================================================

def test_layernorm_different_feature_sizes():
    for feature_size in [1, 2, 4, 8, 32, 64]:

        x = make_input((4, feature_size))

        ln = nn.LayerNorm(feature_size)

        y = ln(x)

        assert y.shape == x.shape


def test_layernorm_single_feature():
    x = torch.Tensor(
        np.array([
            [1.0],
            [2.0],
            [3.0],
        ])
    )

    ln = nn.LayerNorm(1)

    y = ln(x)

    # With only one feature:
    # mean = x
    # variance = 0
    # therefore normalized value = 0
    np.testing.assert_allclose(
        y.numpy(),
        np.zeros((3, 1)),
        atol=1e-5,
    )


# ============================================================
# Parameter registration
# ============================================================

def test_layernorm_named_parameters():
    ln = nn.LayerNorm(16)

    names = [
        name
        for name, _ in ln.named_parameters()
    ]

    assert "weight" in names
    assert "bias" in names


def test_layernorm_parameters():
    ln = nn.LayerNorm(16)

    parameters = list(ln.parameters())

    assert len(parameters) == 2


# ============================================================
# Backward / Autograd
# ============================================================

def test_layernorm_backward():
    x = make_input((4, 8))
    x.requires_grad = True

    ln = nn.LayerNorm(8)

    y = ln(x)

    loss = y.sum()

    loss.backward()

    assert x.grad is not None
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def test_layernorm_backward_shapes():
    x = make_input((4, 8))
    x.requires_grad = True

    ln = nn.LayerNorm(8)

    y = ln(x)

    loss = y.sum()

    loss.backward()

    assert x.grad.shape == x.shape
    assert ln.weight.grad.shape == (8,)
    assert ln.bias.grad.shape == (8,)


def test_layernorm_gradients_are_finite():
    x = make_input((4, 8))
    x.requires_grad = True

    ln = nn.LayerNorm(8)

    y = ln(x)

    loss = (y * y).sum()

    loss.backward()

    assert np.all(np.isfinite(x.grad))
    assert np.all(np.isfinite(ln.weight.grad))
    assert np.all(np.isfinite(ln.bias.grad))


# ============================================================
# Train / Eval
# ============================================================

def test_layernorm_train_eval_same_output():
    x = make_input((4, 8))

    ln = nn.LayerNorm(8)

    ln.train()
    y_train = ln(x)

    ln.eval()
    y_eval = ln(x)

    # Unlike BatchNorm, LayerNorm does not use
    # running statistics, so train/eval should give
    # the same result.
    np.testing.assert_allclose(
        y_train.numpy(),
        y_eval.numpy(),
        atol=1e-6,
    )


# ============================================================
# Input should not be modified
# ============================================================

def test_layernorm_does_not_modify_input():
    x_np = np.random.randn(4, 8)
    x = torch.Tensor(x_np.copy())

    ln = nn.LayerNorm(8)

    _ = ln(x)

    np.testing.assert_array_equal(
        x.numpy(),
        x_np,
    )