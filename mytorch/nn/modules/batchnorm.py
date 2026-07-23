from .module import Module
from ..parameter import Parameter
import numpy as np

class _BatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        self.affine = affine
        
        self.weight = self.bias = None
        if affine:
            self.weight = Parameter(np.ones(num_features))
            self.bias = Parameter(np.zeros(num_features))
        
        self.running_mean = self.running_var = None
        if track_running_stats:
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
        
    def forward(self, x):
        reduce_dims = self._get_reduce_dims(x)
        mean = var = None
        if not self.training and self.track_running_stats:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = x.mean(axis=reduce_dims)
            var = ((x - self._broadcast(mean, x)) ** 2).mean(axis=reduce_dims)
         
        broadcast_mean = self._broadcast(mean, x)
        broadcast_var = self._broadcast(var, x)
        x_norm = (x - broadcast_mean) / ((broadcast_var + self.eps) ** 0.5)
        
        if self.training and self.track_running_stats:
            
            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean
            )

            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * var
            )
        
        if not self.affine:
            return x_norm
        
        weight = self._broadcast(self.weight, x)
        bias = self._broadcast(self.bias, x)
        
        return weight * x_norm + bias
        
        
class BatchNorm1d(_BatchNorm):
    def _get_reduce_dims(self, x):
        if x.ndim == 2:
            return (0,)
        elif x.ndim == 3:
            return (0, 2)
        
    def _broadcast(self, tensor, x):
        if x.ndim == 2:
            return tensor.reshape(1, -1)
        elif x.ndim == 3:
            return tensor.reshape(1, -1, 1)

class BatchNorm2d(_BatchNorm):
    def _get_reduce_dims(self, x):
        return (0, 2, 3)
    
    def _broadcast(self, tensor, x):
        return tensor.reshape(1, -1, 1, 1)

class BatchNorm3d(_BatchNorm):
    def _get_reduce_dims(self, x):
        return (0, 2, 3, 4)
    
    def _broadcast(self, tensor, x):
        return tensor.reshape(1, -1, 1, 1, 1)