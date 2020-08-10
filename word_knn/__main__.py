import sys
from word_knn.nlpl_retriever import from_nlpl
import argparse
from pathlib import Path
from flask import Flask
from flask_restful import Resource, Api

home = str(Path.home())


def main():
    parser = argparse.ArgumentParser(description="Find closest words.")
    parser.add_argument("--word", help="word", type=str, default="cat")
    parser.add_argument("--count", help="number of nearest neighboors", type=int, default=10)
    parser.add_argument("--with_distances", help="display distance to words", type=bool, default=False)
    parser.add_argument(
        "--keep_embeddings_in_memory",
        help="keep embeddings in memory (useful for non reconstructible index)",
        type=bool,
        default=False,
    )
    parser.add_argument("--root_embeddings_dir", help="dir to save embeddings", type=str, default=home + "/embeddings")
    parser.add_argument(
        "--embeddings_id", help="word embeddings id from http://vectors.nlpl.eu/repository/", type=str, default="0"
    )
    parser.add_argument("--save_zip", help="save the zip (default false)", default=False, type=bool)
    parser.add_argument("--serve", help="serve http API to get nearest words", default=False, type=bool)

    args = parser.parse_args()

    closest_words = from_nlpl(
        args.root_embeddings_dir, args.embeddings_id, args.save_zip, args.keep_embeddings_in_memory
    )
    if not args.serve:
        print(closest_words.closest_words(args.word, args.count, args.keep_embeddings_in_memory))
    else:
        app = Flask(__name__)
        api = Api(app)

        class NearestWords(Resource):
            def get(self, word, count, with_distances):
                return closest_words.closest_words(
                    word, count, with_distances == "distance", args.keep_embeddings_in_memory
                )

        class NearestWordsSimple(Resource):
            def get(self, word, count=10):
                return closest_words.closest_words(word, count, False)

        class NearestWordsSimpleCount(Resource):
            def get(self, word, count):
                return closest_words.closest_words(word, count, False)

        api.add_resource(NearestWordsSimple, "/<string:word>")
        api.add_resource(NearestWordsSimpleCount, "/<string:word>/<int:count>")
        api.add_resource(NearestWords, "/<string:word>/<int:count>/<string:with_distances>")
        print("Go to http://127.0.0.1:5000/dog/10")
        app.run(host="0.0.0.0")


if __name__ == "__main__":
    sys.exit(main())
