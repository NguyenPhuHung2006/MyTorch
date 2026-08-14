from mytorch.datasets import DataLoader, Dataset, MNIST
import mytorch as torch
import mytorch.nn as nn
import mytorch.optim as optim
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, dataset: MNIST):
        images = dataset.images
        labels = dataset.labels
        images = np.expand_dims(images, axis=1)
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
            # 28x28
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 13x13
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 5x5
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),

            # 5x5x128 = 3200
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.layers(x)
    
EPOCHS = 5
criterion = nn.CrossEntropyLoss()
model = MNISTSolver()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

from tqdm import tqdm

for epoch in range(EPOCHS):
    
    t_cce = 0
    model.train()
    
    train_pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]",
        unit="batch",
    )
    
    for imgs, labels in train_pbar:
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t_cce += loss.item()
        
        # Show running average loss in the progress bar
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_t_cce = t_cce / len(train_loader)
    
    # ---------------- Validation ----------------
    v_cce = 0
    total_acc = 0
    model.eval()
    
    val_pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
        unit="batch",
    )
    
    for imgs, labels in val_pbar:
        logits = model(imgs)
        preds = np.argmax(logits.numpy(), axis=-1)
        mask = (labels == preds).numpy()
        total_acc += mask.sum() / len(mask)
        
        loss = criterion(logits, labels)
        v_cce += loss.item()
        
        val_pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_v_cce = v_cce / len(val_loader)
    avg_acc = total_acc / len(val_loader)
    
    print(
        f"Epoch: {epoch + 1}/{EPOCHS} | "
        f"Train: {avg_t_cce:.6f} | "
        f"Val: {avg_v_cce:.6f} | "
        f"Acc: {avg_acc * 100:.4f}%"
    )
      
from PIL import Image
import os  

base = "./mytorch/examples/outputs/mnist_cnn/mnist_cnn"
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

print(f"Results written to: {output_dir}")

