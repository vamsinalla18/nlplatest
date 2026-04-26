import torch
import pandas as pd
from tokenizers import Tokenizer

from model.multitask_model import MultiTaskModel
from training.triples_dataset import TriplesDataset
from evaluation.link_prediction import link_prediction
from evaluation.visualize_embeddings import visualize_embeddings


def load_model(model_path="model.pt"):
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

    return model, device


model, device = load_model()

dataset = TriplesDataset(
    "data/triples.csv",
    "tokenizer/medical_tokenizer.json"
)

link_prediction(model, dataset, device)

visualize_embeddings(model, "tokenizer/medical_tokenizer.json", device)
