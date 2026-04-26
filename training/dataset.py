import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
import random


class MLMDataset(Dataset):

    def __init__(self, corpus_file, tokenizer_path, max_len=128):

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        with open(corpus_file, "r") as f:
            self.lines = [l.strip() for l in f if l.strip()]

        self.max_len = max_len

        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.lines)

    def mask_tokens(self, input_ids):

        labels = [-100] * len(input_ids)

        for i in range(len(input_ids)):

            prob = random.random()

            if prob < 0.15:

                labels[i] = input_ids[i]

                prob /= 0.15

                if prob < 0.8:
                    input_ids[i] = self.mask_token_id

                elif prob < 0.9:
                    input_ids[i] = random.randint(0, self.tokenizer.get_vocab_size() - 1)

                else:
                    pass

        return input_ids, labels

    def __getitem__(self, idx):

        text = self.lines[idx]

        encoding = self.tokenizer.encode(text)

        input_ids = encoding.ids

        input_ids = input_ids[:self.max_len]

        input_ids, labels = self.mask_tokens(input_ids)

        padding = self.max_len - len(input_ids)

        input_ids = input_ids + [self.pad_token_id] * padding
        labels = labels + [-100] * padding

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        }