"""Pre-download every model the app needs so container cold starts don't hit the network."""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app import config


def main():
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import pipeline

    SentenceTransformer("all-MiniLM-L6-v2")
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    CrossEncoder(config.NLI_MODEL)
    pipeline("text-classification", model=config.JAILBREAK_MODEL)
    print("models cached")


if __name__ == "__main__":
    main()
