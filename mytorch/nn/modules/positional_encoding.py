from .module import Module
from ...tensor import Tensor
from .dropout import Dropout
import numpy as np

class PositionalEncoding(Module):
    def __init__(
        self,
        d_model,
        max_seq_len=5000,
        dropout=0.0,
    ):
        super().__init__()

        position = np.arange(max_seq_len)[:, None]
        even_dims = np.arange(0, d_model, 2)
        odd_dims = np.arange(1, d_model, 2)
        
        div_term = np.exp(
            -np.log(10000.0) * even_dims / d_model
        )
        
        pe = np.empty((max_seq_len, d_model))
        pe[:, even_dims] = np.sin(position * div_term)
        pe[:, odd_dims] = np.cos(position * div_term[:len(odd_dims)])

        self.pe = pe
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor):
        seq_len = x.shape[1]

        x = x + self.pe[:seq_len]

        return self.dropout(x)