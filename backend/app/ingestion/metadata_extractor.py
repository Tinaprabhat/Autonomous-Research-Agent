import json
import os

METADATA_DIR = "data/metadata"

def save_metadata(papers):

    os.makedirs(METADATA_DIR, exist_ok=True)

    filepath = os.path.join(METADATA_DIR, "papers_metadata.json")

    with open(filepath, "w") as f:
        json.dump(papers, f, indent=4)

    print("Metadata saved:", filepath)