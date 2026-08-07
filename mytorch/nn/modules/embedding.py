import numpy as np

from .module import Module
from ..parameter import Parameter

class Embedding(Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(
            np.random.randn(num_embeddings, embedding_dim)
        )

    def forward(self, x):
        return self.weight[x]