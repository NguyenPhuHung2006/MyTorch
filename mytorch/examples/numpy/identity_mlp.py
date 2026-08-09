import mytorch.nn as nn
import mytorch as torch
import mytorch.optim as optim
import numpy as np

# python -m mytorch.examples.identity

x = np.arange(100, dtype=np.float32).reshape(-1, 1)
x /= 99

y = x.copy()

x = torch.Tensor(x)
y = torch.Tensor(y)

class IdentitySolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.layer(x)
    
EPOCHS = 10000
criterion = nn.MSELoss()
# model = IdentitySolver()
# optimizer = optim.SGD(model.parameters(), lr=1e-3)
# for i in range(EPOCHS):
#     optimizer.zero_grad()
    
#     preds = model(x)
#     loss = criterion(preds, y)
    
#     loss.backward()    
#     optimizer.step()
    
#     print(f"{i}, {loss.data}")
    
# preds = model(x).data
# errors = np.abs(preds - y.data)

# print(f"max: {errors.max()}")
# print(f"mean: {errors.mean()}")

RUN = 10
for run in range(RUN):
    model = IdentitySolver()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    model.zero_grad()
    for i in range(EPOCHS):
        optimizer.zero_grad()
    
        logits = model(x)
        loss = criterion(logits, y)
        
        loss.backward()    
        optimizer.step()
        
    preds = model(x).data
    errors = np.abs(preds - y.data)
    print(f"{run}, max: {errors.max():.5}, min: {errors.min():.5}")


