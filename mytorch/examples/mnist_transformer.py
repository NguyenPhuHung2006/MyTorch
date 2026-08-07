from mytorch.datasets import DataLoader, Dataset, MNIST
import mytorch as torch
import mytorch.nn as nn
import mytorch.optim as optim
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, dataset: MNIST):
        images = dataset.images
        labels = dataset.labels
        images = images.astype(np.float32) / 255.0
        
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
train_dataset = MNIST("./data", train=True)
test_dataset = MNIST("./data", train=False)

train_dataset = MNISTDataset(train_dataset)
test_dataset = MNISTDataset(test_dataset)

train_loader = DataLoader(train_dataset)
val_loader = DataLoader(test_dataset)

D_MODEL = 32
N_HEAD = 4
N_LAYERS = 1
D_FF = 64
DROPOUT = 0.1

class MNISTTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(28, D_MODEL)

        self.positional_encoding = nn.PositionalEncoding(
            d_model=D_MODEL,
            max_seq_len=28,
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=N_LAYERS,
        )

        self.classifier = nn.Linear(D_MODEL, 10)

    def forward(self, x):
        # x: (B, 28, 28)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)

        x = x.mean(axis=1)

        return self.classifier(x)
        
    
EPOCHS = 20
criterion = nn.CrossEntropyLoss()
model = MNISTTransformer()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for i in range(EPOCHS):
    
    t_cce = 0
    model.train()
    
    for imgs, labels in train_loader:
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t_cce += loss.item()
        
    avg_t_cce = t_cce / len(train_loader)
    
    v_cce = 0
    total_acc = 0
    model.eval()
    for imgs, labels in val_loader:
        logits = model(imgs)
        preds = np.argmax(logits.numpy(), axis=-1)
        mask = (labels == preds).numpy()
        total_acc += mask.sum() / len(mask)
        
        loss = criterion(logits, labels)
        
        v_cce += loss.item()
        
    avg_v_cce = v_cce / len(val_loader)
    avg_acc = total_acc / len(val_loader)
    
    print(f"Epochs: {i+1}/{EPOCHS} | Train: {avg_t_cce:.6} | Val: {avg_v_cce:.6} | Acc: {avg_acc * 100:.4}%")
      
from PIL import Image
import os  

base = "./mytorch/examples/outputs/mnist_transformer/mnist_transformer"
i = 1

while os.path.exists(f"{base}_{i}"):
    i += 1

output_dir = f"{base}_{i}"
os.makedirs(output_dir)

model.eval()

idx = 0

for imgs, labels in val_loader:
    logits = model(imgs)

    preds = np.argmax(logits.numpy(), axis=-1)

    imgs = imgs.numpy()
    labels = labels.numpy()

    for image, pred, label in zip(imgs, preds, labels):
        if pred != label:
            img = (image.reshape(28, 28) * 255).astype(np.uint8)

            filename = f"{idx}_true_{label}_pred_{pred}.png"
            Image.fromarray(img).save(
                os.path.join(output_dir, filename)
            )

            idx += 1


