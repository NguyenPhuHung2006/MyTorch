from .module import Module
from ...tensor import Tensor
from ..parameter import Parameter
from .container import ModuleList
import mytorch as torch
from .dropout import Dropout
from .. import init
import numpy as np

class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = Parameter(np.empty((4 * hidden_size, input_size)))
        self.weight_hh = Parameter(np.empty((4 * hidden_size, hidden_size)))
        
        self.bias = None
        if bias:
            self.bias = Parameter(np.empty(4 * hidden_size))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        bound = 1 / np.sqrt(self.hidden_size)
        init.uniform_(self.weight_ih, -bound, bound)
        init.uniform_(self.weight_hh, -bound, bound)
        
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x: Tensor, h: Tensor | None = None, c: Tensor | None = None):
        if x.ndim != 2:
            raise ValueError(
                f"LSTMCell expected x to be 2D (batch, input_size), "
                f"but got shape {x.shape}."
            )

        if x.shape[1] != self.input_size:
            raise ValueError(
                f"LSTMCell expected input_size={self.input_size}, "
                f"but got x.shape={x.shape}."
            )

        if h is None:
            h = Tensor(np.zeros((x.shape[0], self.hidden_size)))
        if c is None:
            c = Tensor(np.zeros((x.shape[0], self.hidden_size)))

        if h.ndim != 2:
            raise ValueError(
                f"LSTMCell expected h to be 2D (batch, hidden_size), "
                f"but got shape {h.shape}."
            )

        if h.shape != (x.shape[0], self.hidden_size):
            raise ValueError(
                f"LSTMCell expected hidden state shape "
                f"({x.shape[0]}, {self.hidden_size}), "
                f"but got {h.shape}."
            )
            
        if c.ndim != 2:
            raise ValueError(
                f"LSTMCell expected c to be 2D (batch, hidden_size), "
                f"but got shape {c.shape}."
            )

        if c.shape != (x.shape[0], self.hidden_size):
            raise ValueError(
                f"LSTMCell expected hidden state shape "
                f"({x.shape[0]}, {self.hidden_size}), "
                f"but got {c.shape}."
            )
            
        out_ih = x @ self.weight_ih.T
        out_hh = h @ self.weight_hh.T
        
        out = out_ih + out_hh
        if self.bias is not None:
            out += self.bias
        
        i = out[:, :self.hidden_size].sigmoid()
        f = out[:, self.hidden_size:self.hidden_size*2].sigmoid()
        g = out[:, self.hidden_size*2:self.hidden_size*3].tanh()
        o = out[:, self.hidden_size*3:].sigmoid()
        
        c = f * c + i * g
        h = o * c.tanh()
        return h, c
        
            
class LSTM(Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers=1, 
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
        self.bias = bias
        self.dropout_layer = Dropout(dropout)
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        for layer in range(num_layers):
            in_features = input_size if layer == 0 else hidden_size * self.num_directions
            
            forward_cell = LSTMCell(
                in_features,
                hidden_size,
                bias=bias,
            )
            
            cells = [forward_cell]
            
            if bidirectional:
                backward_cell = LSTMCell(
                    in_features,
                    hidden_size,
                    bias=bias,
                )
                cells.append(backward_cell)
            
            self.cells.append(ModuleList(cells))
            
    def _validate_input(self, x: Tensor):
        if x.ndim != 3:
            raise ValueError(
                f"LSTM expected 3D input, but got shape {x.shape}."
            )

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, input_size = x.shape

        if input_size != self.input_size:
            raise ValueError(
                f"LSTM expected input_size={self.input_size}, "
                f"but got {input_size}."
            )

        return x, seq_len, batch_size
    
    def _prepare_hidden_and_cell(self, h, c, batch_size):
        expected = (
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
        )

        if h is None:
            h = Tensor(np.zeros(expected))
        if c is None:
            c = Tensor(np.zeros(expected))

        if h.ndim != 3:
            raise ValueError(
                f"LSTM expected hidden state to have 3 dimensions, "
                f"but got shape {h.shape}."
            )

        if h.shape != expected:
            raise ValueError(
                f"LSTM expected hidden state shape {expected}, "
                f"but got {h.shape}."
            )
            
        if c.ndim != 3:
            raise ValueError(
                f"LSTM expected cell state to have 3 dimensions, "
                f"but got shape {c.shape}."
            )
    
        if c.shape != expected:
            raise ValueError(
                f"LSTM expected cell state shape {expected}, "
                f"but got {c.shape}."
            )

        return h, c
    
    def _run_direction(self, inputs, cell, h, c, reverse=False):
        seq_len = inputs.shape[0]

        outputs = [None] * seq_len

        time_steps = range(seq_len)
        if reverse:
            time_steps = reversed(range(seq_len))

        for t in time_steps:
            h, c = cell(inputs[t], h, c)
            outputs[t] = h

        return outputs, h, c
    
    def _run_layer(self, inputs, h, c, layer):
        if self.bidirectional:
            return self._run_bidirectional_layer(inputs, h, c, layer)

        outputs, h_last, c_last = self._run_direction(
            inputs,
            self.cells[layer][0],
            h[layer],
            c[layer],
        )

        return torch.stack(outputs), [h_last], [c_last]
    
    def _run_bidirectional_layer(self, inputs, h, c, layer):
        h_forward = h[2 * layer]
        h_backward = h[2 * layer + 1]
        c_forward = c[2 * layer]
        c_backward = c[2 * layer + 1]

        forward_outputs, h_forward, c_forward = self._run_direction(
            inputs,
            self.cells[layer][0],
            h_forward,
            c_forward,
        )

        backward_outputs, h_backward, c_backward = self._run_direction(
            inputs,
            self.cells[layer][1],
            h_backward,
            c_backward,
            reverse=True,
        )

        outputs = [
            torch.cat([f, b], axis=-1)
            for f, b in zip(forward_outputs, backward_outputs)
        ]

        return torch.stack(outputs), [h_forward, h_backward], [c_forward, c_backward]
    
    def _empty_output(self, batch_size):
        outputs = Tensor(
            np.empty((0, batch_size, self.hidden_size * self.num_directions))
        )
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs
            
    def forward(self, x: Tensor, h: Tensor | None = None, c: Tensor | None = None):
        x, seq_len, batch_size = self._validate_input(x)
        h, c = self._prepare_hidden_and_cell(h, c, batch_size)
                
        if seq_len == 0:
            return self._empty_output(batch_size), h, c
        
        inputs = x
        hiddens = []
        cells = []
        for layer in range(self.num_layers):
            inputs, layer_hiddens, layer_cells = self._run_layer(inputs, h, c, layer)
            hiddens.extend(layer_hiddens)
            cells.extend(layer_cells)
            
            if layer < self.num_layers - 1 and self.dropout > 0:
                inputs = self.dropout_layer(inputs)
            
        hiddens = torch.stack(hiddens)
        cells = torch.stack(cells)
        outputs = inputs
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        return outputs, hiddens, cells       