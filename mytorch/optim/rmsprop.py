from .optimizer import Optimizer
import numpy as np

class RMSProp(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
    ):
        super().__init__(params)

        self.lr = lr
        self.alpha = alpha
        self.eps = eps

        self.square_avg = [
            np.zeros_like(p.data)
            for p in self.params
        ]
    
    def step(self):

        for i, param in enumerate(self.params):

            if param.grad is None:
                continue

            grad = param.grad

            self.square_avg[i] = (
                self.alpha * self.square_avg[i]
                + (1 - self.alpha) * grad**2
            )

            param.data -= (
                self.lr
                * grad
                / (np.sqrt(self.square_avg[i]) + self.eps)
            )