from .module import Module
from .container import ModuleList
from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .layernorm import LayerNorm
from .dropout import Dropout

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
        d_model,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
    ):
        super().__init__()

        self.layers = ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x