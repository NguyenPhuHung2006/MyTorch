from ..tensor import Tensor
import numpy as np

class Engine:
    def __init__(self):
        pass
    
    def backward(self, loss: Tensor, init_grad: np.ndarray | None):
        topo = self.sort_topo(loss)
        loss.grad = np.ones_like(loss.data) if init_grad is None else init_grad
        
        for tensor in reversed(topo):
            if tensor.grad_fn is None:
                continue

            grads = tensor.grad_fn.backward(tensor.grad)
            
            for parent, grad in zip(tensor.grad_fn.parents, grads):
                if not isinstance(parent, Tensor) or not parent.requires_grad:
                    continue
                if parent.grad is None:
                    parent.grad = grad.copy()
                else:
                    parent.grad += grad
    
    
    def sort_topo(self, loss: Tensor):
        topo = []
        visited = set()
        
        def dfs(tensor):
            if tensor is None or not isinstance(tensor, Tensor) or tensor in visited:
                return
            visited.add(tensor)
            if tensor.grad_fn is not None:
                for next in tensor.grad_fn.parents:
                    dfs(next)
            topo.append(tensor)
            
        dfs(loss)
        return topo
        