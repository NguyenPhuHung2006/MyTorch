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
        
        
from mytorch.nn import Transformer, Module


# ============================================================
# Helpers
# ============================================================

def make_transformer(
    d_model=16,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=32,
    dropout=0.0,
    batch_first=True,
):
    return Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=batch_first,
    )


# ============================================================
# Basic forward
# ============================================================

def test_transformer_output_shape():
    batch_size = 4
    source_len = 10
    target_len = 6
    d_model = 16

    transformer = make_transformer(
        d_model=d_model,
        batch_first=True,
    )

    src = Tensor(
        np.random.randn(
            batch_size,
            source_len,
            d_model,
        )
    )

    tgt = Tensor(
        np.random.randn(
            batch_size,
            target_len,
            d_model,
        )
    )

    output = transformer(src, tgt)

    assert output.shape == (
        batch_size,
        target_len,
        d_model,
    )


def test_transformer_can_handle_different_source_target_lengths():
    transformer = make_transformer()

    src = Tensor(
        np.random.randn(2, 12, 16)
    )

    tgt = Tensor(
        np.random.randn(2, 5, 16)
    )

    output = transformer(src, tgt)

    assert output.shape == (
        2,
        5,
        16,
    )


def test_transformer_preserves_target_length():
    transformer = make_transformer()

    src = Tensor(
        np.random.randn(3, 20, 16)
    )

    tgt = Tensor(
        np.random.randn(3, 7, 16)
    )

    output = transformer(src, tgt)

    assert output.shape[0] == 3
    assert output.shape[1] == 7
    assert output.shape[2] == 16


# ============================================================
# Number of layers
# ============================================================

def test_transformer_has_correct_number_of_encoder_layers():
    transformer = make_transformer(
        num_encoder_layers=4,
        num_decoder_layers=3,
    )

    assert len(transformer.encoder.layers) == 4


def test_transformer_has_correct_number_of_decoder_layers():
    transformer = make_transformer(
        num_encoder_layers=4,
        num_decoder_layers=3,
    )

    assert len(transformer.decoder.layers) == 3


# ============================================================
# Encoder / Decoder construction
# ============================================================

def test_transformer_creates_encoder():
    transformer = make_transformer()

    assert isinstance(
        transformer.encoder,
        TransformerEncoder,
    )


def test_transformer_creates_decoder():
    transformer = make_transformer()

    assert isinstance(
        transformer.decoder,
        TransformerDecoder,
    )


def test_transformer_encoder_contains_encoder_layers():
    transformer = make_transformer()

    for layer in transformer.encoder.layers:
        assert isinstance(
            layer,
            TransformerEncoderLayer,
        )


def test_transformer_decoder_contains_decoder_layers():
    transformer = make_transformer()

    for layer in transformer.decoder.layers:
        assert isinstance(
            layer,
            TransformerDecoderLayer,
        )


# ============================================================
# Parameter independence
# ============================================================

def test_encoder_layers_are_independent():
    transformer = make_transformer(
        num_encoder_layers=3,
    )

    layer0 = transformer.encoder.layers[0]
    layer1 = transformer.encoder.layers[1]

    assert layer0 is not layer1

    params0 = list(layer0.parameters())
    params1 = list(layer1.parameters())

    assert len(params0) == len(params1)

    for p0, p1 in zip(params0, params1):
        assert p0 is not p1


def test_decoder_layers_are_independent():
    transformer = make_transformer(
        num_decoder_layers=3,
    )

    layer0 = transformer.decoder.layers[0]
    layer1 = transformer.decoder.layers[1]

    assert layer0 is not layer1

    params0 = list(layer0.parameters())
    params1 = list(layer1.parameters())

    assert len(params0) == len(params1)

    for p0, p1 in zip(params0, params1):
        assert p0 is not p1


# ============================================================
# batch_first
# ============================================================

def test_transformer_batch_first_true():
    transformer = make_transformer(
        batch_first=True,
    )

    src = Tensor(
        np.random.randn(2, 10, 16)
    )

    tgt = Tensor(
        np.random.randn(2, 6, 16)
    )

    output = transformer(src, tgt)

    assert output.shape == (
        2,
        6,
        16,
    )


def test_transformer_batch_first_false():
    transformer = make_transformer(
        batch_first=False,
    )

    # External API:
    # (seq_len, batch, d_model)

    src = Tensor(
        np.random.randn(10, 2, 16)
    )

    tgt = Tensor(
        np.random.randn(6, 2, 16)
    )

    output = transformer(src, tgt)

    assert output.shape == (
        6,
        2,
        16,
    )


def test_batch_first_modes_produce_same_shape_after_transpose():
    transformer_batch_first = make_transformer(
        batch_first=True,
    )

    transformer_sequence_first = make_transformer(
        batch_first=False,
    )

    src = np.random.randn(2, 10, 16)
    tgt = np.random.randn(2, 6, 16)

    output_batch_first = transformer_batch_first(
        Tensor(src),
        Tensor(tgt),
    )

    output_sequence_first = transformer_sequence_first(
        Tensor(src.transpose(1, 0, 2)),
        Tensor(tgt.transpose(1, 0, 2)),
    )

    assert output_batch_first.shape == (
        2,
        6,
        16,
    )

    assert output_sequence_first.shape == (
        6,
        2,
        16,
    )


# ============================================================
# Cross-attention
# ============================================================

def test_transformer_output_depends_on_source():
    transformer = make_transformer(
        num_encoder_layers=1,
        num_decoder_layers=1,
    )

    tgt = Tensor(
        np.random.randn(2, 5, 16)
    )

    src1 = Tensor(
        np.random.randn(2, 10, 16)
    )

    src2 = Tensor(
        np.random.randn(2, 10, 16)
    )

    output1 = transformer(src1, tgt)
    output2 = transformer(src2, tgt)

    assert not np.allclose(
        output1.data,
        output2.data,
    )


def test_transformer_output_depends_on_target():
    transformer = make_transformer(
        num_encoder_layers=1,
        num_decoder_layers=1,
    )

    src = Tensor(
        np.random.randn(2, 10, 16)
    )

    tgt1 = Tensor(
        np.random.randn(2, 5, 16)
    )

    tgt2 = Tensor(
        np.random.randn(2, 5, 16)
    )

    output1 = transformer(src, tgt1)
    output2 = transformer(src, tgt2)

    assert not np.allclose(
        output1.data,
        output2.data,
    )


# ============================================================
# Masks
# ============================================================

def test_transformer_accepts_src_mask():
    transformer = make_transformer()

    src = Tensor(
        np.random.randn(2, 8, 16)
    )

    tgt = Tensor(
        np.random.randn(2, 5, 16)
    )

    # Adapt this mask to whatever your attention implementation expects.
    src_mask = np.zeros((8, 8))

    output = transformer(
        src,
        tgt,
        src_mask=src_mask,
    )

    assert output.shape == (
        2,
        5,
        16,
    )


def test_transformer_accepts_tgt_mask():
    transformer = make_transformer()

    src = Tensor(
        np.random.randn(2, 8, 16)
    )

    tgt = Tensor(
        np.random.randn(2, 5, 16)
    )

    # Causal mask shape:
    # (target_len, target_len)
    tgt_mask = np.triu(
        np.ones((5, 5)),
        k=1,
    )

    output = transformer(
        src,
        tgt,
        tgt_mask=tgt_mask,
    )

    assert output.shape == (
        2,
        5,
        16,
    )


def test_transformer_accepts_both_masks():
    transformer = make_transformer()

    src = Tensor(
        np.random.randn(2, 8, 16)
    )

    tgt = Tensor(
        np.random.randn(2, 5, 16)
    )

    src_mask = np.zeros((8, 8))

    tgt_mask = np.triu(
        np.ones((5, 5)),
        k=1,
    )

    output = transformer(
        src,
        tgt,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
    )

    assert output.shape == (
        2,
        5,
        16,
    )


# ============================================================
# Named parameters
# ============================================================

def test_transformer_named_parameters_contain_encoder():
    transformer = make_transformer()

    names = [
        name
        for name, _ in transformer.named_parameters()
    ]

    assert any(
        name.startswith("encoder.")
        for name in names
    )


def test_transformer_named_parameters_contain_decoder():
    transformer = make_transformer()

    names = [
        name
        for name, _ in transformer.named_parameters()
    ]

    assert any(
        name.startswith("decoder.")
        for name in names
    )


# ============================================================
# Backward
# ============================================================

def test_transformer_backward():
    transformer = make_transformer(
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )

    src = Tensor(
        np.random.randn(2, 5, 8),
        requires_grad=True,
    )

    tgt = Tensor(
        np.random.randn(2, 4, 8),
        requires_grad=True,
    )

    output = transformer(src, tgt)

    loss = output.sum()

    loss.backward()

    assert src.grad is not None
    assert tgt.grad is not None


def test_transformer_all_parameters_receive_gradients():
    transformer = make_transformer(
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )

    src = Tensor(
        np.random.randn(2, 5, 8),
        requires_grad=True,
    )

    tgt = Tensor(
        np.random.randn(2, 4, 8),
        requires_grad=True,
    )

    output = transformer(src, tgt)

    output.sum().backward()

    for name, param in transformer.named_parameters():
        assert param.grad is not None, (
            f"No gradient for {name}"
        )


# ============================================================
# Custom encoder / decoder
# ============================================================

class DummyEncoder(Module):
    def forward(self, x, mask=None):
        return x


class DummyDecoder(Module):
    def forward(self, tgt, memory, tgt_mask=None):
        return tgt


def test_transformer_accepts_custom_encoder():
    encoder = DummyEncoder()

    transformer = Transformer(
        d_model=16,
        nhead=4,
        custom_encoder=encoder,
    )

    assert transformer.encoder is encoder


def test_transformer_accepts_custom_decoder():
    decoder = DummyDecoder()

    transformer = Transformer(
        d_model=16,
        nhead=4,
        custom_decoder=decoder,
    )

    assert transformer.decoder is decoder


def test_transformer_uses_custom_encoder_and_decoder():
    transformer = Transformer(
        d_model=16,
        nhead=4,
        custom_encoder=DummyEncoder(),
        custom_decoder=DummyDecoder(),
    )

    src = Tensor(
        np.random.randn(2, 10, 16)
    )

    tgt = Tensor(
        np.random.randn(2, 5, 16)
    )

    output = transformer(src, tgt)

    assert output.shape == (
        2,
        5,
        16,
    )