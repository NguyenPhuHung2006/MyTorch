from .module import Module

class Loss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()

        self._reductions = {
            "none": lambda x: x,
            "mean": lambda x: x.mean(),
            "sum": lambda x: x.sum(),
        }

        if reduction not in self._reductions:
            raise ValueError(f"Invalid reduction: {reduction}")

        self.reduction = reduction

    def reduce(self, loss):
        return self._reductions[self.reduction](loss)
    
class MSELoss(Loss):
    def forward(self, pred, target):
        loss = (pred - target) ** 2
        return self.reduce(loss)
    
class BCELoss(Loss):
    def forward(self, pred, target):
        pass