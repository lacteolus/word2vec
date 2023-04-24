"""
Helper functions
"""
import numpy as np
import string
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
from src.vocab import Vocab
from tqdm.auto import tqdm


def display_pca_scatterplot(model, words: list = None, sample=0) -> None:
    """
    Display scatterplot with words distribution
    :param model: Model to be used
    :param words: List of words to be displayed on the scatterplot
    :param sample: Number of words to display. It's used when 'words' param is None.
    In this case n random words are selected from the dictionary and displayed. If sample=0, all words are used
    :return: None
    """
    if words is None:
        if sample > 0:
            words = np.random.choice(list(model.wv.vocab.keys()), sample)
        else:
            words = [word for word in model.wv.vocab]

    word_vectors = np.array([model[w] for w in words])

    two_dim = PCA().fit_transform(word_vectors)[:, :2]

    plt.figure(figsize=(10, 10))
    plt.scatter(two_dim[:, 0], two_dim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, two_dim):
        plt.annotate(text=word, xy=(x, y))


def clean(inp: str) -> str:
    """
    Cleans text by removing all punctuation and special characters. "Your string!" -> "your string"
    :param inp: Input text as string
    :return: Cleaned text as string
    """
    output = inp.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    output = re.sub(r'\s+', ' ', output.lower())
    return output


def tokenize(inp: str, vocab_size: int, default_token="<unk>") -> (Vocab, list):
    """
    Creates list of tokens (words) from input string. Words should be separated with space.
    :param inp: Input string
    :param vocab_size: Vocabulary size
    :param default_token: Default token
    :return: List of tokens (words)
    """
    exclusions = ["the", "of", "and", "in", "a", "to", "for"]
    tokens = clean(inp.strip()).split(" ")
    # Remove small words and prepositions
    tokens = [token for token in tokens if token not in exclusions and len(token) > 2]
    # Count tokens
    counts = Counter(tokens)
    # Create vocabulary
    vocab_tokens = [token for token, _ in counts.most_common(vocab_size - 1)]
    vocab = Vocab(tokens=vocab_tokens, default_token="<unk>")
    # Final tokens
    f_tokens = []
    for token in tqdm(tokens):
        if token in vocab_tokens:
            f_tokens.append(token)
        else:
            f_tokens.append(default_token)
    return vocab, f_tokens
