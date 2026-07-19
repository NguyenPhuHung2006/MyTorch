from mytorch.datasets import DataLoader, Dataset, MNIST
import mytorch as torch
import mytorch.nn as nn
import mytorch.optim as optim
import numpy as np

def flatten(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1)

class MNISTDataset(Dataset):
    def __init__(self, dataset: MNIST):
        images = dataset.images
        labels = dataset.labels
        images = flatten(images)
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

class MNISTSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(train_dataset.images.shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.layers(x)
    
EPOCHS = 20
criterion = nn.CrossEntropyLoss()
model = MNISTSolver()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for i in range(EPOCHS):
    t_cce = 0
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

base = "./mytorch/examples/outputs/mnist_mlp/mnist_mlp"
i = 1

while os.path.exists(f"{base}_{i}"):
    i += 1

output_dir = f"{base}_{i}"
os.makedirs(output_dir)

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


