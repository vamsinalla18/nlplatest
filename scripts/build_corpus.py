from datasets import load_dataset
import os
from tqdm import tqdm

os.makedirs("data", exist_ok=True)

print("Loading MedQuad medical dataset...")

dataset = load_dataset(
    "keivalya/MedQuad-MedicalQnADataset",
    split="train"
)

sentences = []

for row in tqdm(dataset):

    q = row["Question"]
    a = row["Answer"]

    if len(q) > 20:
        sentences.append(q)

    if len(a) > 20:
        sentences.append(a)

print("Total sentences:", len(sentences))

print("Saving corpus...")

with open("data/corpus.txt", "w") as f:
    for s in sentences:
        f.write(s.strip() + "\n")

print("Corpus saved.")