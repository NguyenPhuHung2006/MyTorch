from torchvision.datasets import MNIST as TorchMNIST
from .dataset import Dataset

class MNIST(Dataset):
    def __init__(self, root, train=True, download=True):
        torch_dataset = TorchMNIST(
            root=root,
            train=train,
            download=download,
        )

        self.images = torch_dataset.data.numpy()
        self.labels = torch_dataset.targets.numpy()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
