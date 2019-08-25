import sys
from word_knn.nlpl_retriever import from_nlpl
import argparse
from pathlib import Path
from flask import Flask
from flask_restful import Resource, Api

home = str(Path.home())


def main():
    parser = argparse.ArgumentParser(description='Find closest words.')
    parser.add_argument('--word', help='word', type=str, default="cat")
    parser.add_argument('--count', help='number of nearest neighboors', type=int, default=10)
    parser.add_argument('--root_embeddings_dir', help='dir to save embeddings', type=str, default=home + "/embeddings")
    parser.add_argument('--embeddings_id', help='word embeddings id from http://vectors.nlpl.eu/repository/', type=str,
                        default="0")
    parser.add_argument('--save_zip', help='save the zip (default false)', default=False, type=bool)
    parser.add_argument('--serve', help='serve http API to get nearest words', default=False, type=bool)

    args = parser.parse_args()

    closest_words = from_nlpl(args.root_embeddings_dir, args.embeddings_id, args.save_zip)
    if not args.serve :
        print(closest_words.closest_words(args.word, args.count))
    else:
        app = Flask(__name__)
        api = Api(app)

        class NearestWords(Resource):
            def get(self, word, count):
                return closest_words.closest_words(word, count)

        api.add_resource(NearestWords, '/<string:word>/<int:count>')
        print("Go to http://127.0.0.1:5000/dog/10")
        app.run()


if __name__ == '__main__':
    sys.exit(main())
