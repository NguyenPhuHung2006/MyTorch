from .module import Module
from ...tensor import Tensor
from ..parameter import Parameter
from .container import ModuleList
from .. import init
import numpy as np

class RNNCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True, nonlinearity='tanh'):
        super().__init__()
        
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(
                f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}"
            )
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = Parameter(np.empty((hidden_size, input_size)))
        self.weight_hh = Parameter(np.empty((hidden_size, hidden_size)))
        
        self.bias = None
        if bias:
            self.bias = Parameter(np.empty(hidden_size))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        bound = 1 / np.sqrt(self.hidden_size)
        init.uniform_(self.weight_ih, -bound, bound)
        init.uniform_(self.weight_hh, -bound, bound)
        
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: Tensor, h: Tensor = None):
        if x.ndim != 2:
            raise ValueError(
                f"RNNCell expected x to be 2D (batch, input_size), "
                f"but got shape {x.shape}."
            )

        if x.shape[1] != self.input_size:
            raise ValueError(
                f"RNNCell expected input_size={self.input_size}, "
                f"but got x.shape={x.shape}."
            )

        if h is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))

        if h.ndim != 2:
            raise ValueError(
                f"RNNCell expected h to be 2D (batch, hidden_size), "
                f"but got shape {h.shape}."
            )

        if h.shape != (x.shape[0], self.hidden_size):
            raise ValueError(
                f"RNNCell expected hidden state shape "
                f"({x.shape[0]}, {self.hidden_size}), "
                f"but got {h.shape}."
            )
            
        x_proj = x @ self.weight_ih.T
        h_proj = h @ self.weight_hh
        
        out = x_proj + h_proj
        if self.bias is not None:
            out += self.bias
            
        return out.tanh()
            
class RNN(Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers=1, 
                 nonlinearity='tanh', 
                 bias=True, 
                 batch_first=False, 
                 dropout=0.0, 
                 bidirectional=False
                ):
        super().__init__()
        
        self.layers = ModuleList()
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            self.layers.append(RNNCell(in_features, hidden_size, bias, nonlinearity))
            
    def forward(self, x: Tensor, h: Tensor | None = None):
        if x.ndim != 3:
            raise ValueError(
                f"RNN expected 3D input, but got shape {x.shape}."
            )
            
        if self.batch_first:
            x = x.transpose(0, 1)
            
        seq_len, batch_size, input_size = x.shape
        
        if input_size != self.input_size:
            raise ValueError(
                f"RNN expected input_size={self.input_size}, "
                f"but got {input_size}."
            )
            
        if h is None:
            h = Tensor(np.zeros((self.num_layers, x.shape[1], self.hidden_size)))
        
        if h.ndim != 3:
            raise ValueError(
                f"RNN expected h to have shape "
                f"(num_layers, batch, hidden_size), "
                f"but got {h.shape}."
            )

        expected = (self.num_layers, batch_size, self.hidden_size)

        if h.shape != expected:
            raise ValueError(
                f"RNN expected hidden state shape {expected}, "
                f"but got {h.shape}."
            )
            
        output = []
        for t in range(seq_len):
            
            for idx_layer in range(self.num_layers):
                input_t = x[t] if idx_layer == 0 else h[idx_layer - 1]
                h[idx_layer] = self.layers[idx_layer](input_t, h[idx_layer])
                
            output.append(h[-1])
            
        return output, h
        
            
        