import torch
import torch.nn as nn

class BERTEmbeddings(nn.Module):

    def __init__(self, vocab_size, hidden_dim=256, max_len=512):

        super().__init__()

        # token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)

        # positional embeddings
        self.position_embeddings = nn.Embedding(max_len, hidden_dim)

        # segment embeddings
        self.segment_embeddings = nn.Embedding(2, hidden_dim)

        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids):

        seq_len = input_ids.size(1)

        positions = torch.arange(seq_len).unsqueeze(0).to(input_ids.device)

        token_embed = self.token_embeddings(input_ids)
        pos_embed = self.position_embeddings(positions)
        seg_embed = self.segment_embeddings(segment_ids)

        x = token_embed + pos_embed + seg_embed

        x = self.layernorm(x)
        x = self.dropout(x)

        return x