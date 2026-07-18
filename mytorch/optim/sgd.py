from .optimizer import Optimizer
import numpy as np

class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.0,
    ):
        super().__init__(params)

        self.lr = lr
        self.momentum = momentum

        self.velocity = None
        if momentum > 0:
            self.velocity = [
                np.zeros_like(p.data)
                for p in self.params
            ]

    def step(self):
        
        for i, param in enumerate(self.params):

            if param.grad is None:
                continue

            grad = param.grad

            if self.momentum == 0:
                param.data -= self.lr * grad
            else:
                self.velocity[i] = (
                    self.momentum * self.velocity[i] + grad
                )
                param.data -= self.lr * self.velocity[i]