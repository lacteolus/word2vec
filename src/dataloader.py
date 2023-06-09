"""
Functions and classes to prepare data, build vocabulary and dataloaders
"""
import torch
import string
import re
from torch.utils.data import Dataset, DataLoader
from functools import partial
from typing import List
from src.vocab import Vocab
from tqdm.auto import tqdm
from collections import Counter

WINDOW_SIZE = 5
TOKENS_CHUNK_SIZE = WINDOW_SIZE * 10


class CustomDataset(Dataset):
    def __init__(self, tokens: list, chunk_size: int = TOKENS_CHUNK_SIZE):
        """
        Custom dataset
        :param tokens: Input text as list of tokens
        :param chunk_size: Chunk size to split the text
        """
        self.text_iter = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    # Mandatory for map-style datasets
    def __len__(self):
        return len(self.text_iter)

    # Mandatory for map-style datasets
    def __getitem__(self, idx: int) -> str:
        """
        For a given index (idx) returns one item from dataset
        :param idx: Index of the item in dataset
        :return: Text in the specified chunk
        """
        return self.text_iter[idx]


def collate_cbow(batch, vocab: Vocab) -> (torch.Tensor, torch.Tensor):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    Context is represented as N=WINDOW_SIZE words before and after the central word.
    Each element in `batch_input` is N=WINDOW_SIZE*2 context words.
    Each element in `batch_output` is a central word.
    :param batch: String of space-separated words.
    :param vocab: Vocabulary used to
    :return: Input and output as torch.Tensor's
    """
    batch_input, batch_output = [], []

    for data in batch:
        tokens_ids = vocab.lookup_indices(data)
        # Iterate through all tokens using sliding window
        for idx in range(len(tokens_ids) - WINDOW_SIZE * 2):
            token_id_sequence = tokens_ids[idx: (idx + WINDOW_SIZE * 2 + 1)]  # Get tokens within a window
            central_token = token_id_sequence.pop(WINDOW_SIZE)  # Get central token
            context_tokens = token_id_sequence  # Get context tokens
            batch_input.append(context_tokens)
            batch_output.append(central_token)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def collate_skipgram(batch, vocab: Vocab) -> (torch.Tensor, torch.Tensor):
    """
    Collate_fn for SkipGram model to be used with Dataloader.
    Context is represented as N=WINDOW_SIZE words before and after the central word.
    Each element in `batch_input` is a central word.
    Each element in `batch_output` is a context word.
    :param batch: String of space-separated words.
    :param vocab: Vocabulary used to
    :return: Input and output as torch.Tensor's
    """
    batch_input, batch_output = [], []

    for data in batch:
        tokens_ids = vocab.lookup_indices(data)
        # Iterate through all tokens using sliding window
        for idx in range(len(tokens_ids) - WINDOW_SIZE * 2):
            token_id_sequence = tokens_ids[idx: (idx + WINDOW_SIZE * 2 + 1)]  # Get tokens within a window
            central_token = token_id_sequence.pop(WINDOW_SIZE)  # Get central token
            context_tokens = token_id_sequence  # Get context tokens

            for item in context_tokens:
                batch_input.append(central_token)
                batch_output.append(item)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def get_dataloader(
        tokens: List[str],
        model_type: str,
        loader_type: str,
        vocab: Vocab,
        batch_size: int = 16
) -> DataLoader:
    """
    Get dataloader and vocabulary
    :param tokens: List of tokens
    :param model_type: Type of the model to be used: either "skipgram" or "cbow
    :param loader_type: Type of the loader: either "train" or "test"
    :param vocab: Vocabulary
    :param batch_size: Dataloader batch size
    :return: Dataloader
    """
    if loader_type == "train":
        text = tokens[:int(0.8 * len(tokens))]  # 80% of text is for training
    elif loader_type == "val":
        text = tokens[int(0.8 * len(tokens)):]  # 20% of text is for testing
    else:
        raise ValueError("Only 'train' and 'val' dataloader types are supported")

    dataset = CustomDataset(text)

    if model_type == "skipgram":
        collate_fn = collate_skipgram
    elif model_type == "cbow":
        collate_fn = collate_cbow
    else:
        raise ValueError("Only 'cbow' and 'skipgram' model types are supported")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, vocab=vocab),
    )


def clean(inp: str) -> str:
    """
    Cleans text by removing all punctuation and special characters. "Your string!" -> "your string"
    :param inp: Input text as string
    :return: Cleaned text as string
    """
    output = inp.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    output = re.sub(r'\s+', ' ', output.lower())
    return output


def tokenize(inp: str, vocab_size: int, default_token="<unk>", min_token_size: int = 3) -> (Vocab, list):
    """
    Creates list of tokens (words) from input string. Words should be separated with space.
    :param inp: Input string
    :param vocab_size: Vocabulary size
    :param default_token: Default token
    :param min_token_size: Minimum token size
    :return: List of tokens (words)
    """
    exclusions = ["the", "of", "and", "in", "a", "to", "for"]
    tokens = clean(inp.strip()).split(" ")
    # Remove small words and prepositions
    tokens = [token for token in tokens if len(token) >= min_token_size and token not in exclusions]
    # Count tokens
    counts = Counter(tokens)
    # Create vocabulary
    vocab_tokens = [token for token, _ in counts.most_common(vocab_size - 1)]
    vocab = Vocab(tokens=vocab_tokens, default_token="<unk>")
    # Final tokens
    f_tokens = []
    stream = tqdm(tokens)
    for token in stream:
        stream.set_description("Tokenizing...")
        if token in vocab_tokens:
            f_tokens.append(token)
        else:
            f_tokens.append(default_token)
    return vocab, f_tokens
