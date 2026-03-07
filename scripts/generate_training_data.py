import json
import os
import random

CHUNK_DIR = "data/processed_chunks"
OUTPUT = "data/training/research_dataset.json"

dataset = []

instructions = [
    "Summarize the research contribution.",
    "Explain the key idea of this research.",
    "What problem does this paper solve?",
    "Describe the proposed method."
]

for file in os.listdir(CHUNK_DIR):

    path = os.path.join(CHUNK_DIR, file)

    with open(path) as f:

        chunks = json.load(f)

        for chunk in chunks:

            example = {
                "instruction": random.choice(instructions),
                "input": chunk,
                "output": chunk[:200]
            }

            dataset.append(example)

os.makedirs("data/training", exist_ok=True)

with open(OUTPUT, "w") as f:
    json.dump(dataset, f, indent=2)

print("Training dataset created:", len(dataset))