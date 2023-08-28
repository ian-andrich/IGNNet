import torch.nn as nn


def embedding_layer(num_embeddings: int, embedding_dim: int = 64):
    return nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
