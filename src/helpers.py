"""
Helper functions
"""
import numpy as np
import string
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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


def tokenize(inp: str) -> list:
    """
    Creates list of tokens (words) from input string. Words should be separated with space.
    :param inp: Input string
    :return: List of tokens (words)
    """
    return clean(inp.strip()).split(" ")
