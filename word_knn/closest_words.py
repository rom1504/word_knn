import faiss
import pickle
import numpy as np


def inverse_dict(d):
    return {v: k for k, v in d.items()}


def build_knn_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


class ClosestWords:
    def __init__(self, embeddings, inverse_word_dict, word_dict, knn_index):
        self.embeddings = embeddings
        self.inverse_word_dict = inverse_word_dict
        self.word_dict = word_dict
        self.knn_index = knn_index

    def closest_words(self, word, k):
        emb = self.embeddings[self.inverse_word_dict[word]]
        words = np.array([emb])
        distances, indices = self.knn_index.search(words, k)
        return [self.word_dict[found_index] for found_index in indices[0]]

    @staticmethod
    def from_disk_cache(cache_dir):
        embeddings = np.load(cache_dir + "/embeddings.npy")
        pickle_in = open(cache_dir + "/word_dict.pkl", "rb")
        word_dict = pickle.load(pickle_in)
        inverse_word_dict = inverse_dict(word_dict)
        knn_index = build_knn_index(embeddings)
        return ClosestWords(embeddings, inverse_word_dict, word_dict, knn_index)

    def cache_to_disk(self, word_embedding_dir):
        np.save(word_embedding_dir + "/embeddings.npy", self.embeddings)
        f = open(word_embedding_dir + "/word_dict.pkl", "wb")
        pickle.dump(self.word_dict, f)
        f.close()
