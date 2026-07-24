from .module import Module
from ...tensor import Tensor
from ...ops.dropout import Dropout as DropoutFunction

class _DropoutNd(Module):
    def __init__(self, p=0.5):
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError(
                f"Dropout probability has to be between 0 and 1, but got {p}."
            )
        self.p = p

class Dropout(_DropoutNd):
    def forward(self, x: Tensor):
        mask_shape = x.shape
        return DropoutFunction.apply(x, self.p, self.training, mask_shape)
    
class Dropout1d(_DropoutNd):
    def forward(self, x: Tensor):
        if x.ndim != 2 and x.ndim != 3:
            raise ValueError(
                f"Dropout1d expected 2D or 3D input, got {x.ndim}D."
            )
        mask_shape = x.shape[:2] + (1,) * (x.ndim - 2)
        return DropoutFunction.apply(x, self.p, self.training, mask_shape)

class Dropout2d(_DropoutNd):
    def forward(self, x: Tensor):
        if x.ndim != 4:
            raise ValueError(
                f"Dropout2d expected 4D input, got {x.ndim}D."
            )
        mask_shape = x.shape[:2] + (1, 1)
        return DropoutFunction.apply(x, self.p, self.training, mask_shape)

class Dropout3d(_DropoutNd):
    def forward(self, x: Tensor):
        if x.ndim != 5:
            raise ValueError(
                f"Dropout3d expected 5D input, got {x.ndim}D."
            )
        mask_shape = x.shape[:2] + (1, 1, 1)
        return DropoutFunction.apply(x, self.p, self.training, mask_shape)