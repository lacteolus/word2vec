"""
Class for building vocabulary.
Implements simplified version of Vocab similar to 'torchtext.vocab.Vocab' class with minimum
required attributes and methods.
"""
from typing import List, Dict
from src.helpers import tokenize


class Vocab:
    def __init__(self, tokens: List[str]):
        """
        :param tokens: List of unique tokens
        """
        self.vocab = tokens

    @classmethod
    def from_text(cls, text):
        """
        Builds vocabulary from the text
        :param text: Cleaned text as a string of space-separated words
        :return: Vocabulary as a list of unique words
        """
        tokens = list(set(tokenize(text)))
        return Vocab(tokens)

    def get_stoi(self) -> Dict[str, int]:
        """
        Get all tokens (words) and their indexes
        :return: Dictionary mapping tokens to indices
        """
        return {item: i for i, item in enumerate(self.vocab)}

    def get_itos(self) -> List[str]:
        """
        Get all tokens (words)
        :return: List mapping indices to tokens
        """
        return self.vocab

    def lookup_token(self, idx: int) -> str:
        """
        Get index by word
        :return: The token used to lookup the corresponding index.
        """
        return self.vocab[idx]

    def lookup_index(self, token: str) -> int:
        """
        Get token (word) by index
        :param token:
        :return: The token used to lookup its corresponding index
        """
        return self.vocab.index(token)

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        """
        Get list of tokens (words) indexes
        :param tokens: List of tokens
        :return: List of indexes. E.g. ["hello", "world"] -> [25, 47]
        """
        return [self.lookup_index(token) for token in tokens]

    def get_size(self) -> int:
        """
        Get the size of the vocabulary
        :return: Size of the vocabulary
        """
        return len(self.vocab)
