from .dataset import Dataset
import numpy as np
from ..tensor import Tensor

class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        self.indices = np.arange(len(self.dataset))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.indices):
            raise StopIteration
        if self.drop_last and self._index + self.batch_size > len(self.indices):
            raise StopIteration
    
        batch_indices = self.indices[self._index:self._index+self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        
        self._index += self.batch_size
        x, y = zip(*batch)
        x = np.stack(x)
        y = np.array(y)
        x = Tensor(x)
        y = Tensor(y)
        return x, y
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size