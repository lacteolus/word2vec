"""
Class for building vocabulary.
Implements simplified version of Vocab similar to 'torchtext.vocab.Vocab' class with minimum
required attributes and methods.
"""
import json
from typing import List, Dict


class Vocab:
    def __init__(self, tokens: List[str], default_token="<unk>"):
        """
        :param tokens: List of unique tokens
        """
        if default_token in tokens:
            self.vocab = tokens
        else:
            self.vocab = [default_token] + tokens
        self.stoi = {item: i for i, item in enumerate(self.vocab)}

    def get_stoi(self) -> Dict[str, int]:
        """
        Get all tokens (words) and their indexes
        :return: Dictionary mapping tokens to indices
        """
        return self.stoi

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

    def __len__(self) -> int:
        """
        Get the size of the vocabulary
        :return: Size of the vocabulary
        """
        return len(self.vocab)

    @classmethod
    def from_file(cls, file_path):
        """
        Download vocabulary from file
        :param file_path: File path
        :return: Vocab object
        """
        with open(file_path, "r") as f:
            vocab_dict = json.load(f)
        return Vocab(list(vocab_dict.keys()))
