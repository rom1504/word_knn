# word-knn
[![Discord](https://img.shields.io/badge/install-from%20conda-brightgreen.svg)](https://anaconda.org/rom1504/word_knn)

Quickly find closest words using an efficient knn and word embeddings

## Installation

First install python3 and [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
then :

```bash
conda install -c pytorch word_knn
```

## Usage

### Command line

Just run `python -m word_knn --word "cat"`

Details :
```bash
$ python -m word_knn --help
usage: python -m word_knn [-h] [--word WORD]
                   [--root_embeddings_dir ROOT_EMBEDDINGS_DIR]
                   [--embeddings_id EMBEDDINGS_ID] [--save_zip SAVE_ZIP]

Find closest words.

optional arguments:
  -h, --help            show this help message and exit
  --word WORD           word
  --root_embeddings_dir ROOT_EMBEDDINGS_DIR
                        dir to save embeddings
  --embeddings_id EMBEDDINGS_ID
                        word embeddings id from
                        http://vectors.nlpl.eu/repository/
  --save_zip SAVE_ZIP   save the zip (default false)

```

### Python interface

First go to http://vectors.nlpl.eu/repository/ and pick some embeddings.
I advise the `Google News 2013` one (id 1).
For these embeddings, you will need about 15GB of disk space and 6GB of RAM.

you can also use id 0 which is smaller
(faster to download) but contains much less words

You can then run this to get some closest words. This will automatically download and extract the embeddings.
```python
from word_knn import from_nlpl
closest_words = from_nlpl("/home/rom1504/embeddings", "0", False)
print(closest_words.closest_words("cat", 10))
```
The word dictionary, embeddings and knn index are then cached. Second run will be much faster.


If you have limited ram, you can also download and extract the embeddings yourself with this :
```
mkdir -p ~/embeddings/0
cd ~/embeddings/0
wget http://vectors.nlpl.eu/repository/11/0.zip
unzip 0.zip
```
```python
from word_knn import from_csv_or_cache
closest_words = from_csv_or_cache("/home/rom1504/embeddings/0")

print(closest_words.closest_words("cat", 10))
```

## Development

### Create an environment

```bash
conda create -n wordknn python=3
conda activate wordknn
conda install faiss-cpu numpy -c pytorch
```

### Rebuild the conda package

run 
```bash
conda build -c pytorch .
```