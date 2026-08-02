from .module import Module
from ...tensor import Tensor
from ..parameter import Parameter
from .container import ModuleList
import mytorch as torch
from .activation import Tanh, ReLU
from .dropout import Dropout
from .. import init
import numpy as np

_ACTIVATIONS = {
    "tanh": Tanh,
    "relu": ReLU,
}

class RNNCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True, nonlinearity='tanh'):
        super().__init__()
        
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(
                f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}"
            )
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.activation = _ACTIVATIONS[nonlinearity]()
        
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
        
    def forward(self, x: Tensor, h: Tensor | None = None):
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
            
        return self.activation(out)
            
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
        
        self.cells = ModuleList()
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.dropout_layer = Dropout(dropout)
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        for layer in range(num_layers):
            in_features = input_size if layer == 0 else hidden_size * self.num_directions
            
            forward_cell = RNNCell(
                in_features,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity
            )
            
            cells = [forward_cell]
            
            if bidirectional:
                backward_cell = RNNCell(
                    in_features,
                    hidden_size,
                    bias=bias,
                    nonlinearity=nonlinearity
                )
                cells.append(backward_cell)
            
            self.cells.append(ModuleList(cells))
            
    def _validate_input(self, x: Tensor):
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

        return x, seq_len, batch_size
    
    def _prepare_hidden(self, h, batch_size):
        expected = (
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
        )

        if h is None:
            return Tensor(np.zeros(expected))

        if h.ndim != 3:
            raise ValueError(
                f"RNN expected h to have 3 dimensions, "
                f"but got shape {h.shape}."
            )

        if h.shape != expected:
            raise ValueError(
                f"RNN expected hidden state shape {expected}, "
                f"but got {h.shape}."
            )

        return h
    
    def _run_direction(self, inputs, cell, h, reverse=False):
        seq_len = inputs.shape[0]

        outputs = [None] * seq_len

        time_steps = range(seq_len)
        if reverse:
            time_steps = reversed(range(seq_len))

        for t in time_steps:
            h = cell(inputs[t], h)
            outputs[t] = h

        return outputs, h
    
    def _run_layer(self, inputs, h, layer):
        if self.bidirectional:
            return self._run_bidirectional_layer(inputs, h, layer)

        outputs, h_last = self._run_direction(
            inputs,
            self.cells[layer][0],
            h[layer],
        )

        return torch.stack(outputs), [h_last]
    
    def _run_bidirectional_layer(self, inputs, h, layer):
        h_forward = h[2 * layer]
        h_backward = h[2 * layer + 1]

        forward_outputs, h_forward = self._run_direction(
            inputs,
            self.cells[layer][0],
            h_forward,
        )

        backward_outputs, h_backward = self._run_direction(
            inputs,
            self.cells[layer][1],
            h_backward,
            reverse=True,
        )

        outputs = [
            torch.cat([f, b], axis=-1)
            for f, b in zip(forward_outputs, backward_outputs)
        ]

        return torch.stack(outputs), [h_forward, h_backward]
    
    def _empty_output(self, batch_size):
        outputs = Tensor(
            np.empty((0, batch_size, self.hidden_size * self.num_directions))
        )
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs
            
    def forward(self, x: Tensor, h: Tensor | None = None):
        x, seq_len, batch_size = self._validate_input(x)
        h = self._prepare_hidden(h, batch_size)
                
        if seq_len == 0:
            return self._empty_output(batch_size), h
        
        inputs = x
        hiddens = []
        for layer in range(self.num_layers):
            inputs, layer_hiddens = self._run_layer(inputs, h, layer)
            hiddens.extend(layer_hiddens)
            
            if layer < self.num_layers - 1 and self.dropout > 0:
                inputs = self.dropout_layer(inputs)
            
        hiddens = torch.stack(hiddens)
        outputs = inputs
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        return outputs, hiddens        