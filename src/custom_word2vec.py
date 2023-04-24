"""
Custom implementations for Word2Vec
"""
import torch.nn as nn


class CBOWModel(nn.Module):
    """
    CBOW model
    """
    def __init__(self, vocab_size: int, embedding_size: int = 100) -> None:
        """
        :param vocab_size: Vocabulary size
        :param embedding_size: Embedding size
        """
        super().__init__()

        # Linear layer
        self.linear = nn.Linear(
            in_features=embedding_size,
            out_features=vocab_size,
        )
        # Embeddings
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            max_norm=1,
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
    def __init__(self, vocab_size: int, embedding_size: int = 100) -> None:
        """
        :param vocab_size: Vocabulary size
        :param embedding_size: Embedding size
        """
        super().__init__()

        # Linear layer
        self.linear = nn.Linear(
            in_features=embedding_size,
            out_features=vocab_size,
        )

        # Embeddings
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            max_norm=1,
        )

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = self.linear(x)
        return x
