from ..tensor import Tensor
from ..nn.modules.sequential import Sequential
from ..nn.modules.module import Module
from ..nn.modules.linear import Linear
from ..nn.modules.activations import ReLU
from ..nn.modules.loss import BCEWithLogitsLoss
from ..optim.sgd import SGD

x = [
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
]

y = [[1], [1], [0], [0]]

x = Tensor(x)
y = Tensor(y)

class XorSolver(Module):
    def __init__(self):
        super().__init__()
        self.layer = Sequential(
            Linear(2, 3),
            ReLU(),
            Linear(3, 3),
            ReLU(),
            Linear(3, 1)
        )
    def forward(self, x):
        return self.layer(x)
    
EPOCHS = 10000
model = XorSolver()
criterion = BCEWithLogitsLoss()
optimizer = SGD(model.parameters(), lr=1e-2)
for i in range(EPOCHS):
    optimizer.zero_grad()
    
    logits = model(x)
    loss = criterion(logits, y)
    
    loss.backward()    
    optimizer.step()
    
    print(f"{i}, {loss.data}")
    
print(model(x).sigmoid().data > 0.5)

