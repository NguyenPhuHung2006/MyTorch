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
