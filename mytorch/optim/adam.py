from .optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        super().__init__(params)

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        self.step_count = 0

        self.exp_avg = [
            np.zeros_like(p.data)
            for p in self.params
        ]

        self.exp_avg_sq = [
            np.zeros_like(p.data)
            for p in self.params
        ]

    def step(self):

        self.step_count += 1

        for i, param in enumerate(self.params):

            if param.grad is None:
                continue

            grad = param.grad

            self.exp_avg[i] = (
                self.beta1 * self.exp_avg[i]
                + (1 - self.beta1) * grad
            )

            self.exp_avg_sq[i] = (
                self.beta2 * self.exp_avg_sq[i]
                + (1 - self.beta2) * grad**2
            )

            m_hat = (
                self.exp_avg[i]
                / (1 - self.beta1**self.step_count)
            )

            v_hat = (
                self.exp_avg_sq[i]
                / (1 - self.beta2**self.step_count)
            )

            param.data -= (
                self.lr
                * m_hat
                / (np.sqrt(v_hat) + self.eps)
            )