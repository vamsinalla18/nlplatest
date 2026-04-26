import torch
from torch.utils.data import Dataset
import pandas as pd
from tokenizers import Tokenizer


class TriplesDataset(Dataset):

    def __init__(self, triples_file, tokenizer_path, max_len=16):

        df = pd.read_csv(triples_file, header=None)

        self.heads = df[0].tolist()
        self.relations = df[1].tolist()
        self.tails = df[2].tolist()

        # relation → id
        self.relation2id = {
            r: i for i, r in enumerate(set(self.relations))
        }

        # triples list
        self.triples = [
            (h, self.relation2id[r], t)
            for h, r, t in zip(self.heads, self.relations, self.tails)
        ]

        # entity set
        self.entities = list(set(self.heads + self.tails))

        # entity types
        types_df = pd.read_csv("data/entity_types.csv", header=None)

        self.entity_types = {
            row[0]: row[1]
            for _, row in types_df.iterrows()
        }

        # tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.max_len = max_len
        self.pad_id = self.tokenizer.token_to_id("[PAD]")

    def encode_entity(self, text):

        encoding = self.tokenizer.encode(text)

        ids = encoding.ids[:self.max_len]

        padding = self.max_len - len(ids)

        ids = ids + [self.pad_id] * padding

        return torch.tensor(ids)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):

        h, r, t = self.triples[idx]

        h_tokens = self.encode_entity(h)
        t_tokens = self.encode_entity(t)

        r = torch.tensor(r)

        # h and t names are returned so the trainer can do filtered negative sampling
        return h_tokens, r, t_tokens, h, t