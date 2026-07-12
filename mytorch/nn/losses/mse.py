from .loss import Loss

class MSELoss(Loss):
    def forward(self, pred, target):
        loss = (pred - target) ** 2
        return self.reduce(loss)