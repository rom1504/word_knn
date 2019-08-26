# word_knn
[![Discord](https://img.shields.io/badge/install-from%20conda-brightgreen.svg)](https://anaconda.org/rom1504/word_knn)

Quickly find closest words using an efficient knn and word embeddings. Uses :
* [faiss](https://github.com/facebookresearch/faiss) for an efficient knn implementation
* [nlpl word embeddings](http://vectors.nlpl.eu/repository/) for quality word embeddings

## Installation

First install python3 and [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
then :

```bash
conda activate base
conda install -c pytorch -c rom1504 -c conda-forge word_knn
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
conda install faiss-cpu numpy flask flask-restful -c pytorch -c conda-forge
```

### Rebuild the conda package

First do a `conda install conda-build anaconda-client`, then :

run 
```bash
conda build -c pytorch -c conda-forge .
```

## FAQ

#### I'm getting `Illegal instruction (core dumped)`

It means your CPU doesn't support some recent instructions.
Install an older version of faiss `conda install faiss-cpu=1.5.1 -c pytorch -y`
For details, see https://github.com/facebookresearch/faiss/issues/885
