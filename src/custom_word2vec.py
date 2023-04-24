"""
Custom implementations for Word2Vec
"""
import torch.nn as nn

EMBEDDING_SIZE = 100  # Embedding (vector) size
EMBEDDING_MAX_NORM = 1  # Maximum normalization


class CBOWModel(nn.Module):
    """
    CBOW model
    """
    def __init__(self, vocab_size: int) -> None:
        """
        :param vocab_size: Vocabulary size
        """
        super().__init__()

        # Linear layer
        self.linear = nn.Linear(
            in_features=EMBEDDING_SIZE,
            out_features=vocab_size,
        )
        # Embeddings
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_SIZE,
            max_norm=EMBEDDING_MAX_NORM,
        )

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGramModel(nn.Module):
    """
    Skip-Gram model
    """
    def __init__(self, vocab_size: int) -> None:
        """
        :param vocab_size: Vocabulary size
        """
        super().__init__()

        # Linear layer
        self.linear = nn.Linear(
            in_features=EMBEDDING_SIZE,
            out_features=vocab_size,
        )

        # Embeddings
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_SIZE,
            max_norm=EMBEDDING_MAX_NORM,
        )

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = self.linear(x)
        return x
