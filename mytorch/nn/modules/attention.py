from .module import Module
from ...tensor import Tensor
from .linear import Linear
from .dropout import Dropout
import numpy as np

class ScaledDotProductAttention(Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        
        self.dropout = Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None):
        d_k = q.shape[-1]

        scores = q @ k.T
        scores = scores / np.sqrt(d_k)

        if mask is not None:
            scores = scores + mask

        weights = scores.softmax(dim=-1)
        weights = self.dropout(weights)

        output = weights @ v

        return output

class SelfAttention(Module):
    def __init__(
        self,
        embed_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(dropout=dropout)

        self.out_proj = Linear(embed_dim, embed_dim)
        
    def forward(self, x: Tensor, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        x = self.attention(q, k, v, mask)

        return self.out_proj(x)