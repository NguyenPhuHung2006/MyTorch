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
    
class MultiHeadAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, but embed_dim = {embed_dim}, num_heads = {num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(
            embed_dim,
            embed_dim,
            bias=bias,
        )

        self.k_proj = Linear(
            embed_dim,
            embed_dim,
            bias=bias,
        )

        self.v_proj = Linear(
            embed_dim,
            embed_dim,
            bias=bias,
        )

        self.attention = ScaledDotProductAttention(
            dropout=dropout,
        )

        self.out_proj = Linear(
            embed_dim,
            embed_dim,
            bias=bias,
        )
        
    def _split_heads(self, x: Tensor):
        B, L, D = x.shape

        x = x.reshape(B, L, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)

        return x
        
    def _combine_heads(self, x: Tensor):
        B, H, L, Dh = x.shape

        x = x.transpose(1, 2)
        x = x.reshape(B, L, H * Dh)

        return x
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask=None):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        x = self.attention(q, k, v, mask)
        
        x = self._combine_heads(x)
        
        x = self.out_proj(x)
        
        return x