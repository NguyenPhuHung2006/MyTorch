from .module import Module
import numpy as np

class Loss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()

        self._REDUCTIONS = {
            "none": lambda x: x,
            "mean": lambda x: x.mean(),
            "sum": lambda x: x.sum(),
        }

        if reduction not in self._REDUCTIONS:
            raise ValueError(f"Invalid reduction: {reduction}")

        self.reduction = reduction

    def reduce(self, loss):
        return self._REDUCTIONS[self.reduction](loss)
    
    @staticmethod
    def _check_same_shape(input, target):
        if input.shape != target.shape:
            raise ValueError(
                f"Expected input and target to have the same shape, "
                f"got {input.shape} and {target.shape}."
            )
            
    @staticmethod
    def _check_cross_entropy_shape(logits, target):
        if logits.ndim != 2:
            raise ValueError(
                f"Expected logits to have shape (N, C), got {logits.shape}."
            )

        if target.ndim != 1:
            raise ValueError(
                f"Expected target to have shape (N,), got {target.shape}."
            )

        if logits.shape[0] != target.shape[0]:
            raise ValueError(
                f"Batch size mismatch: logits has batch size {logits.shape[0]}, "
                f"target has batch size {target.shape[0]}."
            )

        if (target < 0).any() or (target >= logits.shape[1]).any():
            raise ValueError(
                f"Target contains invalid class indices. "
                f"Expected values in [0, {logits.shape[1] - 1}]."
            )
    
    @staticmethod
    def _check_bce_target(target):
        if (target < 0).any() or (target > 1).any():
            raise ValueError(
                "Target values must be in the range [0, 1]."
            )

    @staticmethod
    def _check_bce_probs(probs):
        if (probs < 0).any() or (probs > 1).any():
            raise ValueError(
                "Input probabilities must be in the range [0, 1]. "
                "If your model outputs logits, use BCEWithLogitsLoss instead."
            )
    
class MSELoss(Loss):
    def forward(self, pred, target):
        self._check_same_shape(pred, target)
        
        loss = (pred - target) ** 2
        return self.reduce(loss)
    
class BCELoss(Loss):
    def _bce_loss(self, probs, target):
        loss = -(target * probs.log() +
                 (1 - target) * (1 - probs).log())
        return self.reduce(loss)
    
    def forward(self, probs, target):
        self._check_same_shape(probs, target)
        self._check_bce_target(target)
        self._check_bce_probs(probs)
        
        return self._bce_loss(probs, target)
    
class BCEWithLogitsLoss(BCELoss):
    def forward(self, logits, target):
        self._check_same_shape(logits, target)

        probs = logits.sigmoid()
        return super().forward(probs, target)
        
class CrossEntropyLoss(Loss):
    def forward(self, logits, target):
        self._check_cross_entropy_shape(logits, target)
        
        log_probs = logits.log_softmax(dim=-1)
        batch_indices = np.arange(target.shape[0])
        loss = -log_probs[batch_indices, target]

        return self.reduce(loss)