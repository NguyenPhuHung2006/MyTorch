from .module import Module
from .container import ModuleList
from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .layernorm import LayerNorm
from .dropout import Dropout
from copy import deepcopy

class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )

        self.ffn = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x):
        # Self-attention
        residual = x

        x = self.norm1(x)

        x = self.self_attn(
            x,
            x,
            x,
        )

        x = residual + self.dropout1(x)

        # Feed-forward
        residual = x

        x = self.norm2(x)

        x = self.ffn(x)

        x = residual + self.dropout2(x)

        return x


class TransformerEncoder(Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
    ):
        super().__init__()

        self.layers = ModuleList([
            deepcopy(encoder_layer)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    
class TransformerDecoderLayer(Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)

        self.ffn = FeedForward(
            d_model,
            dim_feedforward,
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
    
    def forward(self, x, memory, tgt_mask=None):
        # Masked self-attention
        residual = x

        x = self.norm1(x)

        x = self.self_attn(
            x,
            x,
            x,
            mask=tgt_mask,
        )

        x = residual + self.dropout1(x)

        # Cross-attention
        residual = x

        x = self.norm2(x)

        x = self.cross_attn(
            x,       # query
            memory,  # key
            memory,  # value
        )

        x = residual + self.dropout2(x)

        # Feed-forward
        residual = x

        x = self.norm3(x)

        x = self.ffn(x)

        x = residual + self.dropout3(x)

        return x
    
class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        
        self.layers = ModuleList([
            deepcopy(decoder_layer)
            for _ in range(num_layers)
        ])

    def forward(self, x, memory, tgt_mask=None):
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
            )

        return x