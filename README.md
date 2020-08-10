# word_knn
[![pypi](https://img.shields.io/pypi/v/word_knn.svg)](https://pypi.python.org/pypi/word_knn)
[![ci](https://github.com/rom1504/word_knn/workflows/Continuous%20integration/badge.svg)](https://github.com/rom1504/word_knn/actions?query=workflow%3A%22Continuous+integration%22)

Quickly find closest words using an efficient knn and word embeddings. Uses :
* [faiss](https://github.com/facebookresearch/faiss) for an efficient knn implementation
* [nlpl word embeddings](http://vectors.nlpl.eu/repository/) for quality word embeddings

## Installation

First install python3
then :

```bash
pip install word_knn
```

## Usage

### Command line

Just run `python -m word_knn --word "cat"`

Details :
```bash
$ python -m word_knn --help
usage: python -m word_knn [-h] [--word WORD] [--count COUNT]
                   [--root_embeddings_dir ROOT_EMBEDDINGS_DIR]
                   [--embeddings_id EMBEDDINGS_ID] [--save_zip SAVE_ZIP]
                   [--serve SERVE]

Find closest words.

optional arguments:
  -h, --help            show this help message and exit
  --word WORD           word
  --count COUNT         number of nearest neighboors
  --root_embeddings_dir ROOT_EMBEDDINGS_DIR
                        dir to save embeddings
  --embeddings_id EMBEDDINGS_ID
                        word embeddings id from
                        http://vectors.nlpl.eu/repository/
  --save_zip SAVE_ZIP   save the zip (default false)
  --serve SERVE         serve http API to get nearest words
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


You can also download and extract the embeddings yourself with this :
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

### Prerequisites

Make sure you use `python>=3.6` and an up-to-date version of `pip` and
`setuptools`

    python --version
    pip install -U pip setuptools

It is recommended to install `word_knn` in a new virtual environment. For
example

    python3 -m venv word_knn_env
    source word_knn_env/bin/activate
    pip install -U pip setuptools
    pip install word_knn

### Using Pip

    pip install word_knn

### From Source

First, clone the `word_knn` repo on your local machine with

    git clone https://github.com/rom1504/word_knn.git
    cd word_knn
    make install

To install development tools and test requirements, run

    make install-dev

## Test

To run unit tests in your current environment, run

    make test

To run lint + unit tests in a fresh virtual environment,
run

    make venv-lint-test

## Lint

To run `black --check`:

    make lint

To auto-format the code using `black`

    make black
