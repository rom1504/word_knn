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

    def closest_words(self, word, k, with_distances=False, use_emb_from_knn_index=True):
        if word not in self.inverse_word_dict:
            return None

        if use_emb_from_knn_index:
            emb = self.knn_index.reconstruct(self.inverse_word_dict[word])
        else:
            emb = self.embeddings[self.inverse_word_dict[word]]
        words = np.array([emb])
        distances, indices = self.knn_index.search(words, k)
        if with_distances:
            return [
                (self.word_dict[found_index], float(distance))
                for found_index, distance in zip(indices[0], distances[0])
            ]
        else:
            return [self.word_dict[found_index] for found_index in indices[0]]

    @staticmethod
    def from_disk_cache(cache_dir, load_embeddings=False, load_knn_index=True):
        if load_embeddings:
            embeddings = np.load(cache_dir + "/embeddings.npy")
        else:
            embeddings = None
        pickle_in = open(cache_dir + "/word_dict.pkl", "rb")
        word_dict = pickle.load(pickle_in)
        inverse_word_dict = inverse_dict(word_dict)
        if load_knn_index:
            knn_index = faiss.read_index(cache_dir + "/knn_index")
        else:
            knn_index = build_knn_index(embeddings)
        return ClosestWords(embeddings, inverse_word_dict, word_dict, knn_index)

    def cache_to_disk(self, cache_dir, save_embeddings=True, save_knn_index=True):
        if save_embeddings:
            np.save(cache_dir + "/embeddings.npy", self.embeddings)
        if save_knn_index:
            faiss.write_index(self.knn_index, cache_dir + "/knn_index")
        f = open(cache_dir + "/word_dict.pkl", "wb")
        pickle.dump(self.word_dict, f)
        f.close()
