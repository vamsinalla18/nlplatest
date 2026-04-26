import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pickle
import pandas as pd
from tokenizers import Tokenizer

from model.multitask_model import MultiTaskModel
from training.type_dataset import TypeDataset


def build_disease_index(model_path="model.pt"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.from_file("tokenizer/medical_tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")

    types_df = pd.read_csv("data/entity_types.csv", header=None)
    num_types = len(sorted(set(types_df[1].tolist())))

    triples_df = pd.read_csv("data/triples.csv", header=None)
    num_relations = len(set(triples_df[1].tolist()))

    model = MultiTaskModel(
        vocab_size,
        num_types=num_types,
        num_relations=num_relations,
        pad_id=pad_id,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = TypeDataset(
        "data/entity_types.csv",
        "tokenizer/medical_tokenizer.json"
    )

    disease_embeddings = {}

    for entity, label in zip(dataset.entities, dataset.types):

        if label != "Disease":
            continue

        tokens = dataset.encode_entity(entity).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.entity_embedding(tokens)

        disease_embeddings[entity] = emb.squeeze(0).cpu()

    with open("disease_index.pkl", "wb") as f:
        pickle.dump(disease_embeddings, f)

    print("Disease index saved")
    print("Total diseases:", len(disease_embeddings))


if __name__ == "__main__":
    build_disease_index()
