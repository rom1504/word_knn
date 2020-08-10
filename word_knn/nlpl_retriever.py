import numpy as np
import os.path
import os
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from word_knn.closest_words import inverse_dict
from word_knn.closest_words import ClosestWords
from word_knn.closest_words import build_knn_index
from pathlib import Path

home = str(Path.home())


def csv_to_embeddings_and_dict(input_file):
    d = dict()

    def read_func(iter):
        next(iter)  # skip first row
        for i, line in enumerate(iter):
            if isinstance(line, str):
                stripped_line = line.rstrip()
            else:
                stripped_line = line.decode("utf-8", "ignore").rstrip()
            line = stripped_line.split(" ")
            word = line[0].split("_")[0]
            d[i] = word.replace("::", " ")
            line.pop(0)
            for item in line:
                yield float(item)
        csv_to_embeddings_and_dict.rowlength = len(line)

    def iter_func():
        csv_to_embeddings_and_dict.rowlength = 0
        if isinstance(input_file, str):
            with open(input_file, "r") as infile:
                yield from read_func(infile)
        else:
            yield from read_func(input_file)

    data = np.fromiter(iter_func(), dtype=float)
    embeddings = data.reshape((-1, csv_to_embeddings_and_dict.rowlength)).astype(np.float32)
    inv_d = inverse_dict(d)

    return embeddings, d, inv_d


def csv_to_dict(input_file):
    d = dict()

    def read(iter):
        next(iter)  # skip first row
        for i, line in enumerate(iter):
            line = line.rstrip().split("_")
            d[i] = line[0].replace("::", " ")

    if isinstance(input_file, str):
        with open(input_file, "r") as infile:
            read(infile)
    else:
        read(input_file)
    inv_d = inverse_dict(d)
    return d, inv_d


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def from_csv(input_file, keep_embeddings=True):
    embeddings, word_dict, inverse_word_dict = csv_to_embeddings_and_dict(input_file)
    knn_index = build_knn_index(embeddings)
    if not keep_embeddings:
        embeddings = None
    return ClosestWords(embeddings, inverse_word_dict, word_dict, knn_index)


def from_csv_or_cache(word_embedding_dir, input_file=None, keep_embeddings=False):
    if input_file is None:
        input_file = word_embedding_dir + "/model.txt"
    if os.path.exists(word_embedding_dir + "/word_dict.pkl"):
        return ClosestWords.from_disk_cache(word_embedding_dir)
    closest_words = from_csv(input_file, True)
    closest_words.cache_to_disk(word_embedding_dir)
    if not keep_embeddings:
        del closest_words.embeddings
    return closest_words


def from_nlpl(root_word_embedding_dir=home + "/embeddings", embedding_id="0", save_zip=False, keep_embeddings=False):
    word_embedding_dir = root_word_embedding_dir + "/" + embedding_id
    if not os.path.exists(word_embedding_dir):
        os.makedirs(word_embedding_dir)

    if os.path.exists(word_embedding_dir + "/word_dict.pkl"):
        return ClosestWords.from_disk_cache(word_embedding_dir, keep_embeddings)

    zip_file_path = word_embedding_dir + "/model.zip"

    if not os.path.exists(word_embedding_dir + "/model.txt"):
        if os.path.exists(zip_file_path):
            zipfile = ZipFile(zip_file_path, "r")
        else:
            url = "http://vectors.nlpl.eu/repository/11/" + embedding_id + ".zip"

            resp = urlopen(url)
            length = resp.getheader("content-length")
            print("Downloading " + url + " (" + sizeof_fmt(int(length)) + ")")
            content = resp.read()
            del resp
            if save_zip:
                file = open(word_embedding_dir + "/model.zip", "wb")
                file.write(content)
                file.close()
            the_bytes = BytesIO(content)
            zipfile = ZipFile(the_bytes)
            del content
            del the_bytes
        zipfile.extract("model.txt", word_embedding_dir)
        zipfile.close()

    return from_csv_or_cache(word_embedding_dir, open(word_embedding_dir + "/model.txt", "rb"), keep_embeddings)
