# tests/nn/test_transformer.py

import numpy as np

from mytorch import Tensor
from mytorch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)


# ============================================================
# Helpers
# ============================================================

def make_encoder_layer(
    d_model=16,
    nhead=4,
    dim_feedforward=32,
    dropout=0.0,
):
    return TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )


def make_decoder_layer(
    d_model=16,
    nhead=4,
    dim_feedforward=32,
    dropout=0.0,
):
    return TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )


# ============================================================
# TransformerEncoderLayer
# ============================================================

def test_encoder_layer_output_shape():
    batch_size = 4
    seq_len = 10
    d_model = 16

    layer = make_encoder_layer(d_model=d_model)

    x = Tensor(
        np.random.randn(batch_size, seq_len, d_model)
    )

    output = layer(x)

    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    )


def test_encoder_layer_preserves_sequence_length():
    layer = make_encoder_layer(d_model=32)

    x = Tensor(
        np.random.randn(3, 7, 32)
    )

    output = layer(x)

    assert output.shape[1] == 7


def test_encoder_layer_different_inputs_produce_different_outputs():
    layer = make_encoder_layer(d_model=16)

    x1 = Tensor(
        np.random.randn(2, 8, 16)
    )

    x2 = Tensor(
        np.random.randn(2, 8, 16)
    )

    output1 = layer(x1)
    output2 = layer(x2)

    assert not np.allclose(
        output1.data,
        output2.data,
    )


def test_encoder_layer_named_parameters():
    layer = make_encoder_layer()

    names = [
        name
        for name, _ in layer.named_parameters()
    ]

    assert any("self_attn" in name for name in names)
    assert any("ffn" in name for name in names)

    assert any("norm1" in name for name in names)
    assert any("norm2" in name for name in names)


# ============================================================
# TransformerEncoder
# ============================================================

def test_encoder_output_shape():
    batch_size = 4
    seq_len = 10
    d_model = 16

    encoder = TransformerEncoder(
        make_encoder_layer(d_model=d_model),
        num_layers=3,
    )

    x = Tensor(
        np.random.randn(
            batch_size,
            seq_len,
            d_model,
        )
    )

    output = encoder(x)

    assert output.shape == x.shape


def test_encoder_has_correct_number_of_layers():
    encoder = TransformerEncoder(
        make_encoder_layer(),
        num_layers=4,
    )

    assert len(encoder.layers) == 4


def test_encoder_layers_are_independent():
    encoder = TransformerEncoder(
        make_encoder_layer(),
        num_layers=3,
    )

    assert encoder.layers[0] is not encoder.layers[1]
    assert encoder.layers[1] is not encoder.layers[2]


def test_encoder_parameters_are_not_shared():
    encoder = TransformerEncoder(
        make_encoder_layer(),
        num_layers=3,
    )

    params0 = list(
        encoder.layers[0].parameters()
    )

    params1 = list(
        encoder.layers[1].parameters()
    )

    assert len(params0) == len(params1)

    for p0, p1 in zip(params0, params1):
        assert p0 is not p1


def test_encoder_named_parameters_include_layer_indices():
    encoder = TransformerEncoder(
        make_encoder_layer(),
        num_layers=2,
    )

    names = [
        name
        for name, _ in encoder.named_parameters()
    ]

    assert any(
        name.startswith("layers.0")
        for name in names
    )

    assert any(
        name.startswith("layers.1")
        for name in names
    )


# ============================================================
# TransformerDecoderLayer
# ============================================================

def test_decoder_layer_output_shape():
    batch_size = 4
    target_len = 6
    source_len = 10
    d_model = 16

    layer = make_decoder_layer(d_model=d_model)

    target = Tensor(
        np.random.randn(
            batch_size,
            target_len,
            d_model,
        )
    )

    memory = Tensor(
        np.random.randn(
            batch_size,
            source_len,
            d_model,
        )
    )

    output = layer(target, memory)

    assert output.shape == (
        batch_size,
        target_len,
        d_model,
    )


def test_decoder_layer_can_handle_different_source_target_lengths():
    batch_size = 2
    target_len = 5
    source_len = 12
    d_model = 16

    layer = make_decoder_layer(
        d_model=d_model,
        nhead=4,
    )

    target = Tensor(
        np.random.randn(
            batch_size,
            target_len,
            d_model,
        )
    )

    memory = Tensor(
        np.random.randn(
            batch_size,
            source_len,
            d_model,
        )
    )

    output = layer(target, memory)

    assert output.shape == (
        batch_size,
        target_len,
        d_model,
    )


def test_decoder_layer_preserves_target_length():
    layer = make_decoder_layer(d_model=32)

    target = Tensor(
        np.random.randn(3, 7, 32)
    )

    memory = Tensor(
        np.random.randn(3, 15, 32)
    )

    output = layer(target, memory)

    assert output.shape[1] == 7


def test_decoder_layer_named_parameters():
    layer = make_decoder_layer()

    names = [
        name
        for name, _ in layer.named_parameters()
    ]

    assert any("self_attn" in name for name in names)
    assert any("cross_attn" in name for name in names)
    assert any("ffn" in name for name in names)

    assert any("norm1" in name for name in names)
    assert any("norm2" in name for name in names)
    assert any("norm3" in name for name in names)


# ============================================================
# Cross-attention
# ============================================================

def test_decoder_output_depends_on_memory():
    batch_size = 2
    target_len = 5
    source_len = 10
    d_model = 16

    layer = make_decoder_layer(
        d_model=d_model,
        nhead=4,
    )

    target = Tensor(
        np.random.randn(
            batch_size,
            target_len,
            d_model,
        )
    )

    memory1 = Tensor(
        np.random.randn(
            batch_size,
            source_len,
            d_model,
        )
    )

    memory2 = Tensor(
        np.random.randn(
            batch_size,
            source_len,
            d_model,
        )
    )

    output1 = layer(target, memory1)
    output2 = layer(target, memory2)

    assert not np.allclose(
        output1.data,
        output2.data,
    )


# ============================================================
# TransformerDecoder
# ============================================================

def test_decoder_output_shape():
    batch_size = 4
    target_len = 6
    source_len = 10
    d_model = 16

    decoder = TransformerDecoder(
        make_decoder_layer(d_model=d_model),
        num_layers=3,
    )

    target = Tensor(
        np.random.randn(
            batch_size,
            target_len,
            d_model,
        )
    )

    memory = Tensor(
        np.random.randn(
            batch_size,
            source_len,
            d_model,
        )
    )

    output = decoder(target, memory)

    assert output.shape == (
        batch_size,
        target_len,
        d_model,
    )


def test_decoder_has_correct_number_of_layers():
    decoder = TransformerDecoder(
        make_decoder_layer(),
        num_layers=4,
    )

    assert len(decoder.layers) == 4


def test_decoder_layers_are_independent():
    decoder = TransformerDecoder(
        make_decoder_layer(),
        num_layers=3,
    )

    assert decoder.layers[0] is not decoder.layers[1]
    assert decoder.layers[1] is not decoder.layers[2]


def test_decoder_parameters_are_not_shared():
    decoder = TransformerDecoder(
        make_decoder_layer(),
        num_layers=3,
    )

    params0 = list(
        decoder.layers[0].parameters()
    )

    params1 = list(
        decoder.layers[1].parameters()
    )

    assert len(params0) == len(params1)

    for p0, p1 in zip(params0, params1):
        assert p0 is not p1


def test_decoder_named_parameters_include_layer_indices():
    decoder = TransformerDecoder(
        make_decoder_layer(),
        num_layers=2,
    )

    names = [
        name
        for name, _ in decoder.named_parameters()
    ]

    assert any(
        name.startswith("layers.0")
        for name in names
    )

    assert any(
        name.startswith("layers.1")
        for name in names
    )


# ============================================================
# Encoder + Decoder integration
# ============================================================

def test_encoder_decoder_shapes():
    batch_size = 2
    source_len = 10
    target_len = 6
    d_model = 16

    encoder = TransformerEncoder(
        make_encoder_layer(d_model=d_model),
        num_layers=2,
    )

    decoder = TransformerDecoder(
        make_decoder_layer(d_model=d_model),
        num_layers=2,
    )

    source = Tensor(
        np.random.randn(
            batch_size,
            source_len,
            d_model,
        )
    )

    target = Tensor(
        np.random.randn(
            batch_size,
            target_len,
            d_model,
        )
    )

    memory = encoder(source)

    output = decoder(
        target,
        memory,
    )

    assert memory.shape == (
        batch_size,
        source_len,
        d_model,
    )

    assert output.shape == (
        batch_size,
        target_len,
        d_model,
    )


# ============================================================
# Backward
# ============================================================

def test_encoder_layer_backward():
    layer = make_encoder_layer(
        d_model=8,
        nhead=2,
    )

    x = Tensor(
        np.random.randn(2, 5, 8),
        requires_grad=True,
    )

    output = layer(x)

    loss = output.sum()

    loss.backward()

    assert x.grad is not None


def test_decoder_layer_backward():
    layer = make_decoder_layer(
        d_model=8,
        nhead=2,
    )

    target = Tensor(
        np.random.randn(2, 5, 8),
        requires_grad=True,
    )

    memory = Tensor(
        np.random.randn(2, 7, 8),
        requires_grad=True,
    )

    output = layer(
        target,
        memory,
    )

    loss = output.sum()

    loss.backward()

    assert target.grad is not None
    assert memory.grad is not None


def test_encoder_all_parameters_receive_gradients():
    layer = make_encoder_layer(
        d_model=8,
        nhead=2,
    )

    x = Tensor(
        np.random.randn(2, 5, 8),
        requires_grad=True,
    )

    output = layer(x)

    output.sum().backward()

    for name, param in layer.named_parameters():
        assert param.grad is not None, (
            f"No gradient for {name}"
        )


def test_decoder_all_parameters_receive_gradients():
    layer = make_decoder_layer(
        d_model=8,
        nhead=2,
    )

    target = Tensor(
        np.random.randn(2, 5, 8),
        requires_grad=True,
    )

    memory = Tensor(
        np.random.randn(2, 7, 8),
        requires_grad=True,
    )

    output = layer(
        target,
        memory,
    )

    output.sum().backward()

    for name, param in layer.named_parameters():
        assert param.grad is not None, (
            f"No gradient for {name}"
        )