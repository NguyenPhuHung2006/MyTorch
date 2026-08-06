from .module import Module
from .linear import Linear
from .dropout import Dropout
from ...tensor import Tensor
from .activation import ReLU, GELU

_ACTIVATIONS = {
    "gelu": GELU,
    "relu": ReLU,
}

class FeedForward(Module):
    def __init__(
        self,
        d_model,
        d_ff,
        dropout=0.0,
        activation="gelu",
    ):
        super().__init__()

        self.linear1 = Linear(d_model, d_ff)
        self.activation = _ACTIVATIONS[activation]()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ff, d_model)
        
    def forward(self, x: Tensor):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x