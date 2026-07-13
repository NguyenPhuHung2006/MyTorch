import mytorch.nn as nn
import mytorch as torch
import mytorch.optim as optim

x = [
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
]

y = [[1], [1], [0], [0]]

x = torch.Tensor(x)
y = torch.Tensor(y)

class XorSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
    def forward(self, x):
        return self.layer(x)
    
EPOCHS = 10000
model = XorSolver()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
for i in range(EPOCHS):
    optimizer.zero_grad()
    
    logits = model(x)
    loss = criterion(logits, y)
    
    loss.backward()    
    optimizer.step()
    
    print(f"{i}, {loss.data}")
    
print(model(x).sigmoid().data > 0.5)

