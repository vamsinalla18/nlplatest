import torch.nn as nn
from model.embeddings import BERTEmbeddings
from model.transformer import TransformerEncoder


class BERTEncoder(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        hidden_dim = 256

        self.embeddings = BERTEmbeddings(vocab_size, hidden_dim)

        self.encoder = TransformerEncoder(
            hidden_dim=hidden_dim,
            heads=8,
            layers=6
        )

    def forward(self, input_ids, segment_ids, padding_mask=None):

        x = self.embeddings(input_ids, segment_ids)

        x = self.encoder(x, mask=padding_mask)

        return x
