"""
Gensim Word2Vec model
"""
import gensim.downloader as api
from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class GensimWord2Vec:
    def __init__(self, dataset_name: str = "text8"):
        """
        :param dataset_name: Dataset used for building vocabulary. "text8" is used by default
        """
        self.dataset = api.load(dataset_name)
        self.model = Word2Vec(self.dataset)

    def show_info(self) -> None:
        """
        Displays basic info about model
        :return:
        """
        print(f"Epochs: {self.model.epochs}")
        print(f"Window size: {self.model.window}")
        print(f"Corpus size: {self.model.corpus_count}")
        print(f"Vector size: {self.model.vector_size}")
        print(f"Vocabulary size: {len(self.model.wv)}")

    def analogy(self, x1, x2, y1) -> str:
        """
        Builds analogy based on word embeddings. E.g.:
            x1 = "king"
            x2 = "man"
            y1 = "queen"
            Existing relationship: "king" -> "man". Returned relationship: "queen" -> "woman"
        :param x1: Negative word
        :param x2: Positive word #1
        :param y1: Positive word #2
        :return: Found word
        """
        return self.model.wv.most_similar(positive=[y1, x2], negative=[x1])[0][0]

    def most_similar(self, word) -> list:
        """
        Find most similar words for the given one
        :param word: Word to be handled
        :return: List of most similar words
        """
        return self.model.wv.most_similar(word)

    def display_pca_scatterplot(self, words: list = None, sample=0) -> None:
        """
        Display scatterplot with words distribution
        :param words: List of words to be displayed on the scatterplot
        :param sample: Number of words to display. It's used when 'words' param is None.
        In this case n random words are selected from the dictionary and displayed. If sample=0, all words are used
        :return: None
        """
        if words is None:
            if sample > 0:
                words = np.random.choice(list(self.model.wv.vocab.keys()), sample)
            else:
                words = [word for word in self.model.wv.vocab]

        word_vectors = np.array([self.model.wv.key_to_index[w] for w in words])

        two_dim = PCA().fit_transform(word_vectors)[:, :2]

        plt.figure(figsize=(10, 10))
        plt.scatter(two_dim[:, 0], two_dim[:, 1], edgecolors='k', c='r')
        for word, (x, y) in zip(words, two_dim):
            plt.annotate(text=word, xy=(x, y))
