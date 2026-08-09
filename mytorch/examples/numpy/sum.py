import mytorch.nn as nn
import mytorch as torch
import mytorch.optim as optim
import numpy as np

# python -m mytorch.examples.sum

N = 100
SEQ_LEN = 100
NUM_LAYERS = 2
HIDDEN_SIZE = 16
DROPOUT = 0
NUM_CLASSES = 10

x = np.random.randint(0, 2, size=(N, SEQ_LEN, 1)).astype(np.float64)
y = (x.sum(axis=1) % NUM_CLASSES).astype(np.int32)
y = np.squeeze(y, axis=-1)

x = torch.Tensor(x)
y = torch.Tensor(y)

class SumSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.proj = nn.Linear(
            out_features=NUM_CLASSES,
            in_features=HIDDEN_SIZE
        )
    def forward(self, x):
        if isinstance(self.rnn, nn.LSTM):
            _, h_n, _ = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        
        h = h_n[-1]
        out = self.proj(h)
        
        return out
        
EPOCHS = 500
criterion = nn.CrossEntropyLoss()

model = SumSolver()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
for i in range(EPOCHS):
    optimizer.zero_grad()
    
    logits = model(x)
    loss = criterion(logits, y)
    
    loss.backward()
    optimizer.step()
    
    preds = np.argmax(logits.numpy(), axis=-1)
    mask = (y == preds).numpy()
    accuracy = mask.sum() / len(mask)

    print(
        f"epoch={i+1}, "
        f"loss={loss.data:.4f}, "
        f"accuracy={accuracy:.2%}"
    )

# RUN = 30
# for run in range(RUN):
#     model = SumSolver()
#     optimizer = optim.Adam(model.parameters(), lr=1e-2)
#     model.zero_grad()
#     for i in range(EPOCHS):
#         optimizer.zero_grad()
    
#         logits = model(x)
#         loss = criterion(logits, y)
        
#         loss.backward()    
#         optimizer.step()
        
#     logits = model(x)
    
#     preds = np.argmax(logits.numpy(), axis=-1)
#     mask = (y == preds).numpy()
#     accuracy = mask.sum() / len(mask)
    
#     print(
#         f"run={run+1}, "
#         f"accuracy={accuracy:.2%}"
#     )
    


