"""
Gensim Word2Vec model
"""
import gensim.downloader as api
from gensim.models import Word2Vec


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
        print(f"Vocabulary size: {len(self.model.wv.vocab)}")

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
        return self.model.most_similar(positive=[y1, x2], negative=[x1])[0][0]
