import torch
from torch.utils.data import Dataset
import pandas as pd
from tokenizers import Tokenizer


class TypeDataset(Dataset):

    def __init__(self, file_path, tokenizer_path, max_len=16):

        df = pd.read_csv(file_path, header=None)

        self.entities = df[0].tolist()
        self.types = df[1].tolist()

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.type2id = {t: i for i, t in enumerate(sorted(set(self.types)))}

        self.max_len = max_len

        self.pad_id = self.tokenizer.token_to_id("[PAD]")

    def encode_entity(self, text):

        encoding = self.tokenizer.encode(text)

        ids = encoding.ids[:self.max_len]

        padding = self.max_len - len(ids)

        ids = ids + [self.pad_id] * padding

        return torch.tensor(ids)

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, idx):

        entity_tokens = self.encode_entity(self.entities[idx])

        label = torch.tensor(self.type2id[self.types[idx]])

        return entity_tokens, label