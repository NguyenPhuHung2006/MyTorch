import numpy as np

class Node:
    def __init__(self, function, ctx, parents=()):
        self.function = function
        self.ctx = ctx
        self.parents = parents
        
    def backward(self, grad_output: np.ndarray):
        return self.function.backward(self.ctx, grad_output)