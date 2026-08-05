import numpy as np
import mytorch as torch
import mytorch.nn as nn


# ============================================================
# Helpers
# ============================================================

def make_input(batch_size=2, seq_len=4, embed_dim=8):
    np.random.seed(42)
    return torch.Tensor(
        np.random.randn(batch_size, seq_len, embed_dim)
    )


# ============================================================
# ScaledDotProductAttention
# ============================================================

def test_scaled_dot_product_attention_output_shape():
    attention = nn.ScaledDotProductAttention(dropout=0.0)

    q = make_input(2, 4, 8)
    k = make_input(2, 4, 8)
    v = make_input(2, 4, 8)

    output = attention(q, k, v)

    assert output.shape == (2, 4, 8)


def test_scaled_dot_product_attention_manual():
    """
    Verify the attention calculation against a NumPy implementation.

    Attention(Q,K,V) =
        softmax(QK^T / sqrt(d_k)) @ V
    """
    attention = nn.ScaledDotProductAttention(dropout=0.0)

    q_np = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    k_np = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    v_np = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
    ])

    q = torch.Tensor(q_np)
    k = torch.Tensor(k_np)
    v = torch.Tensor(v_np)

    output = attention(q, k, v)

    # --------------------------------------------------------
    # Reference implementation
    # --------------------------------------------------------

    d_k = q_np.shape[-1]

    scores = q_np @ k_np.T
    scores = scores / np.sqrt(d_k)

    exp_scores = np.exp(
        scores - np.max(scores, axis=-1, keepdims=True)
    )

    weights = exp_scores / np.sum(
        exp_scores,
        axis=-1,
        keepdims=True
    )

    expected = weights @ v_np

    np.testing.assert_allclose(
        output.numpy(),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_scaled_dot_product_attention_mask():
    """
    Verify that masked positions receive zero attention.

    Causal mask:

        0   -inf -inf
        0    0   -inf
        0    0    0
    """

    attention = nn.ScaledDotProductAttention(dropout=0.0)

    q = torch.Tensor(np.eye(3))
    k = torch.Tensor(np.eye(3))

    v = torch.Tensor(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]))

    mask = torch.Tensor(np.array([
        [0.0, -np.inf, -np.inf],
        [0.0,  0.0,     -np.inf],
        [0.0,  0.0,      0.0],
    ]))

    output = attention(q, k, v, mask=mask)

    # We mainly check that the output is finite and has
    # the expected shape. The exact values depend on your
    # Tensor/softmax implementation.
    assert output.shape == (3, 3)
    assert np.all(np.isfinite(output.numpy()))


def test_scaled_dot_product_attention_zero_dropout():
    """
    With dropout=0, the attention output should be deterministic.
    """

    attention = nn.ScaledDotProductAttention(dropout=0.0)

    q = make_input()
    k = make_input()
    v = make_input()

    output1 = attention(q, k, v)
    output2 = attention(q, k, v)

    np.testing.assert_allclose(
        output1.numpy(),
        output2.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


# ============================================================
# SelfAttention
# ============================================================

def test_self_attention_output_shape():
    attention = nn.SelfAttention(
        embed_dim=8,
        dropout=0.0,
    )

    x = make_input(
        batch_size=2,
        seq_len=5,
        embed_dim=8,
    )

    output = attention(x)

    assert output.shape == (2, 5, 8)


def test_self_attention_different_sequence_lengths():
    attention = nn.SelfAttention(
        embed_dim=8,
        dropout=0.0,
    )

    for seq_len in [1, 2, 5, 10]:
        x = make_input(
            batch_size=2,
            seq_len=seq_len,
            embed_dim=8,
        )

        output = attention(x)

        assert output.shape == (2, seq_len, 8)


def test_self_attention_mask():
    attention = nn.SelfAttention(
        embed_dim=8,
        dropout=0.0,
    )

    x = make_input(
        batch_size=2,
        seq_len=4,
        embed_dim=8,
    )

    mask = torch.Tensor(np.array([
        [0.0, -np.inf, -np.inf, -np.inf],
        [0.0,  0.0,     -np.inf, -np.inf],
        [0.0,  0.0,      0.0,    -np.inf],
        [0.0,  0.0,      0.0,     0.0],
    ]))

    output = attention(x, mask=mask)

    assert output.shape == (2, 4, 8)
    assert np.all(np.isfinite(output.numpy()))


def test_self_attention_batch_independence():
    """
    Changing one sample in the batch should not affect another
    sample because self-attention operates independently across
    the batch dimension.
    """

    attention = nn.SelfAttention(
        embed_dim=8,
        dropout=0.0,
    )

    np.random.seed(42)

    x1_np = np.random.randn(2, 4, 8)
    x2_np = x1_np.copy()

    x2_np[1] += 100.0

    x1 = torch.Tensor(x1_np)
    x2 = torch.Tensor(x2_np)

    y1 = attention(x1)
    y2 = attention(x2)

    # First sample should be identical.
    np.testing.assert_allclose(
        y1.numpy()[0],
        y2.numpy()[0],
        rtol=1e-5,
        atol=1e-5,
    )


def test_self_attention_backward():
    """
    Basic gradient-flow test.

    The important thing here is that backward() completes
    successfully and produces gradients.
    """

    attention = nn.SelfAttention(
        embed_dim=8,
        dropout=0.0,
    )

    x = make_input(
        batch_size=2,
        seq_len=4,
        embed_dim=8,
    )

    x.requires_grad = True

    output = attention(x)

    loss = output.sum()

    loss.backward()

    assert x.grad is not None

    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None, (
            f"Missing gradient for {name}"
        )


import numpy as np
import pytest

import mytorch as torch
import mytorch.nn as nn


# ============================================================
# Helpers
# ============================================================

def make_input(batch_size=2, seq_len=4, embed_dim=8, seed=42):
    rng = np.random.default_rng(seed)
    return torch.Tensor(
        rng.standard_normal((batch_size, seq_len, embed_dim))
    )


# ============================================================
# _split_heads / _combine_heads
# ============================================================

def test_split_heads_shape():
    mha = nn.MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )

    x = make_input(
        batch_size=2,
        seq_len=5,
        embed_dim=8,
    )

    y = mha._split_heads(x)

    # (B, L, D) -> (B, H, L, Dh)
    assert y.shape == (2, 2, 5, 4)


def test_combine_heads_shape():
    mha = nn.MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )

    x = make_input(
        batch_size=2,
        seq_len=5,
        embed_dim=8,
    )

    heads = mha._split_heads(x)
    output = mha._combine_heads(heads)

    assert output.shape == x.shape


def test_split_and_combine_heads_are_inverse():
    mha = nn.MultiHeadAttention(
        embed_dim=12,
        num_heads=3,
        dropout=0.0,
    )

    x = make_input(
        batch_size=2,
        seq_len=7,
        embed_dim=12,
    )

    heads = mha._split_heads(x)
    reconstructed = mha._combine_heads(heads)

    np.testing.assert_allclose(
        reconstructed.numpy(),
        x.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


# ============================================================
# Constructor
# ============================================================

def test_multihead_attention_invalid_embed_dim():
    """
    embed_dim must be divisible by num_heads.
    """

    with pytest.raises(ValueError):
        nn.MultiHeadAttention(
            embed_dim=10,
            num_heads=3,
        )


def test_multihead_attention_head_dim():
    mha = nn.MultiHeadAttention(
        embed_dim=12,
        num_heads=3,
    )

    assert mha.embed_dim == 12
    assert mha.num_heads == 3
    assert mha.head_dim == 4


# ============================================================
# Basic self-attention
# ============================================================

def test_multihead_attention_self_attention_shape():
    """
    Self-attention:

        query = key = value = x

    Input:
        (B, L, D)

    Output:
        (B, L, D)
    """

    mha = nn.MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )

    x = make_input(
        batch_size=2,
        seq_len=5,
        embed_dim=8,
    )

    output = mha(x, x, x)

    assert output.shape == (2, 5, 8)


def test_multihead_attention_different_batch_sizes():
    mha = nn.MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )

    for batch_size in [1, 2, 4, 8]:
        x = make_input(
            batch_size=batch_size,
            seq_len=5,
            embed_dim=8,
        )

        output = mha(x, x, x)

        assert output.shape == (
            batch_size,
            5,
            8,
        )


def test_multihead_attention_different_sequence_lengths():
    mha = nn.MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )

    for seq_len in [1, 2, 5, 10]:
        x = make_input(
            batch_size=2,
            seq_len=seq_len,
            embed_dim=8,
        )

        output = mha(x, x, x)

        assert output.shape == (
            2,
            seq_len,
            8,
        )


# ============================================================
# Different number of heads
# ============================================================

def test_multihead_attention_different_num_heads():
    """
    Different numbers of heads should preserve the final
    embedding dimension.
    """

    embed_dim = 12

    for num_heads in [1, 2, 3, 4, 6, 12]:
        mha = nn.MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
        )

        x = make_input(
            batch_size=2,
            seq_len=5,
            embed_dim=embed_dim,
        )

        output = mha(x, x, x)

        assert output.shape == (
            2,
            5,
            embed_dim,
        )


# ============================================================
# Cross-attention
# ============================================================

def test_multihead_attention_cross_attention():
    """
    Query and key/value can have different sequence lengths.

    query:
        (B, Lq, D)

    key/value:
        (B, Lk, D)

    output:
        (B, Lq, D)
    """

    mha = nn.MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )

    query = make_input(
        batch_size=2,
        seq_len=3,
        embed_dim=8,
        seed=1,
    )

    key = make_input(
        batch_size=2,
        seq_len=7,
        embed_dim=8,
        seed=2,
    )

    value = make_input(
        batch_size=2,
        seq_len=7,
        embed_dim=8,
        seed=3,
    )

    output = mha(
        query,
        key,
        value,
    )

    assert output.shape == (2, 3, 8)


def test_multihead_attention_cross_attention_different_lengths():
    mha = nn.MultiHeadAttention(
        embed_dim=12,
        num_heads=3,
        dropout=0.0,
    )

    query = make_input(
        batch_size=2,
        seq_len=4,
        embed_dim=12,
        seed=1,
    )

    key = make_input(
        batch_size=2,
        seq_len=9,
        embed_dim=12,
        seed=2,
    )

    value = make_input(
        batch_size=2,
        seq_len=9,
        embed_dim=12,
        seed=3,
    )

    output = mha(query, key, value)

    assert output.shape == (2, 4, 12)


# ============================================================
# Mask
# ============================================================

def test_multihead_attention_causal_mask():
    """
    Test that MultiHeadAttention accepts a causal mask.

        0   -inf -inf
        0    0   -inf
        0    0    0
    """

    mha = nn.MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )

    x = make_input(
        batch_size=2,
        seq_len=3,
        embed_dim=8,
    )

    mask = torch.Tensor(
        np.array([
            [0.0, -np.inf, -np.inf],
            [0.0,  0.0,     -np.inf],
            [0.0,  0.0,      0.0],
        ])
    )

    output = mha(
        x,
        x,
        x,
        mask=mask,
    )

    assert output.shape == (2, 3, 8)
    assert np.all(np.isfinite(output.numpy()))


# ============================================================
# Determinism
# ============================================================

def test_multihead_attention_deterministic_without_dropout():
    """
    dropout=0 should make repeated forward passes produce
    the same result.
    """

    mha = nn.MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )

    x = make_input(
        batch_size=2,
        seq_len=5,
        embed_dim=8,
    )

    output1 = mha(x, x, x)
    output2 = mha(x, x, x)

    np.testing.assert_allclose(
        output1.numpy(),
        output2.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


# ============================================================
# Output should depend on input
# ============================================================

def test_multihead_attention_output_changes_with_input():
    mha = nn.MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )

    x1 = make_input(
        batch_size=2,
        seq_len=5,
        embed_dim=8,
        seed=1,
    )

    x2 = make_input(
        batch_size=2,
        seq_len=5,
        embed_dim=8,
        seed=2,
    )

    y1 = mha(x1, x1, x1)
    y2 = mha(x2, x2, x2)

    assert not np.allclose(
        y1.numpy(),
        y2.numpy(),
    )


# ============================================================
# Batch independence
# ============================================================

def test_multihead_attention_batch_independence():
    """
    Attention should operate independently on each batch item.

    Changing sample 1 must not affect sample 0.
    """

    mha = nn.MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )

    rng = np.random.default_rng(42)

    x1_np = rng.standard_normal((2, 5, 8))
    x2_np = x1_np.copy()

    # Change only the second sample.
    x2_np[1] += 100.0

    x1 = torch.Tensor(x1_np)
    x2 = torch.Tensor(x2_np)

    y1 = mha(x1, x1, x1)
    y2 = mha(x2, x2, x2)

    # First batch item must remain unchanged.
    np.testing.assert_allclose(
        y1.numpy()[0],
        y2.numpy()[0],
        rtol=1e-5,
        atol=1e-5,
    )


# ============================================================
# Gradient flow
# ============================================================

def test_multihead_attention_backward():
    """
    Verify that gradients flow through:

        input
          ↓
        Q/K/V
          ↓
       attention
          ↓
       out_proj
    """

    mha = nn.MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )

    x = make_input(
        batch_size=2,
        seq_len=4,
        embed_dim=8,
    )

    x.requires_grad = True

    output = mha(x, x, x)

    loss = output.sum()

    loss.backward()

    assert x.grad is not None

    for name, parameter in mha.named_parameters():
        assert parameter.grad is not None, (
            f"Missing gradient for parameter: {name}"
        )


# ============================================================
# Projection dimensions
# ============================================================

def test_multihead_attention_projection_shapes():
    mha = nn.MultiHeadAttention(
        embed_dim=16,
        num_heads=4,
        dropout=0.0,
    )

    assert mha.q_proj.weight.shape == (16, 16)
    assert mha.k_proj.weight.shape == (16, 16)
    assert mha.v_proj.weight.shape == (16, 16)
    assert mha.out_proj.weight.shape == (16, 16)


# ============================================================
# Single-head should still work
# ============================================================

def test_multihead_attention_single_head():
    """
    MultiHeadAttention with num_heads=1 should still work.

    In this case:

        head_dim = embed_dim
    """

    mha = nn.MultiHeadAttention(
        embed_dim=8,
        num_heads=1,
        dropout=0.0,
    )

    x = make_input(
        batch_size=2,
        seq_len=5,
        embed_dim=8,
    )

    output = mha(x, x, x)

    assert output.shape == (2, 5, 8)
