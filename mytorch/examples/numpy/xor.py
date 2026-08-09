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
            # nn.Linear(2, 3),
            # nn.ReLU(),
            # nn.Linear(3, 3),
            # nn.ReLU(),
            # nn.Linear(3, 1),
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.layer(x)
    
EPOCHS = 10000
criterion = nn.BCEWithLogitsLoss()
# model = XorSolver()
# optimizer = optim.SGD(model.parameters(), lr=1e-2)
# for i in range(EPOCHS):
#     optimizer.zero_grad()
    
#     logits = model(x)
#     loss = criterion(logits, y)
    
#     loss.backward()    
#     optimizer.step()
    
#     print(f"{i}, {loss.data}")
    
RUN = 10
for run in range(RUN):
    model = XorSolver()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    model.zero_grad()
    for i in range(EPOCHS):
        optimizer.zero_grad()
    
        logits = model(x)
        loss = criterion(logits, y)
        
        loss.backward()    
        optimizer.step()
    print(f"{run}, {loss.data:.5}")

