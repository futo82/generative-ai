import argparse

from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(
    prog='similarity_score_cli.py',
    description='This program that takes in 2 text message and generates a cosine similarity score.')
parser.add_argument('--embedding_model', help='The embedding model to use.')
args = parser.parse_args()

EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
if args.embedding_model:
    print("Setting EMBEDDING_MODEL to %s ..." % args.embedding_model)
    EMBEDDING_MODEL = args.embedding_model

# Calculate the cosine similarity between 2 vectors
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def repl() -> None:
    try:
        model = SentenceTransformer(EMBEDDING_MODEL) # TODO: Parameterize the embedding model
        while True:
            try:
                text1 = input("Enter text #1 >> ")
                text2 = input("Enter text #2 >> ")
                embeddings = model.encode([text1, text2])
                score = cosine_similarity(embeddings[0], embeddings[1])
                print()
                print("Similarity Score: %f" % score)
                print()
            except Exception as e:
                print(f"Error: {e}")
    except KeyboardInterrupt as e:
        print("\nExiting...")

if __name__ == "__main__":
    print()
    print("Welcome to the Similar Score CLI")
    print("crtl-c to quit")
    print()
    repl()