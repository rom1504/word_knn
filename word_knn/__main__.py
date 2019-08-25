import sys
from word_knn.nlpl_retriever import from_nlpl
import argparse
from pathlib import Path

home = str(Path.home())


def main():
    parser = argparse.ArgumentParser(description='Find closest words.')
    parser.add_argument('--word', help='word', type=str, default="cat")
    parser.add_argument('--count', help='number of nearest neighboors', type=int, default=10)
    parser.add_argument('--root_embeddings_dir', help='dir to save embeddings', type=str, default=home + "/embeddings")
    parser.add_argument('--embeddings_id', help='word embeddings id from http://vectors.nlpl.eu/repository/', type=str,
                        default="0")
    parser.add_argument('--save_zip', help='save the zip (default false)', default=False, type=bool)

    args = parser.parse_args()

    closestWords = from_nlpl(args.root_embeddings_dir, args.embeddings_id, args.save_zip)
    print(closestWords.closest_words(args.word, args.count))


if __name__ == '__main__':
    sys.exit(main())
