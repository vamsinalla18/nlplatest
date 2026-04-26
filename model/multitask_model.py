import torch
import torch.nn as nn

from model.bert_encoder import BERTEncoder
from heads.mlm_head import MLMHead
from heads.type_head import TypeHead


class MultiTaskModel(nn.Module):

    def __init__(self, vocab_size, hidden_dim=256, num_types=2, num_relations=1, pad_id=0):

        super().__init__()

        self.pad_id = pad_id

        self.encoder = BERTEncoder(vocab_size)

        self.mlm_head = MLMHead(hidden_dim, vocab_size)

        self.type_head = TypeHead(hidden_dim, num_types)

        self.relation_embeddings = nn.Embedding(num_relations, hidden_dim)

    def _padding_mask(self, input_ids):
        # True = ignore this position (PAD). Shape: (B, L)
        return input_ids == self.pad_id

    def _masked_mean(self, hidden, input_ids):
        # Mean-pool only over real (non-PAD) token positions.
        real = (input_ids != self.pad_id).float().unsqueeze(-1)  # (B, L, 1)
        summed = (hidden * real).sum(dim=1)                       # (B, H)
        count = real.sum(dim=1).clamp(min=1e-9)                   # (B, 1)
        return summed / count

    def forward_mlm(self, input_ids, segment_ids):

        pad_mask = self._padding_mask(input_ids)

        hidden = self.encoder(input_ids, segment_ids, padding_mask=pad_mask)

        logits = self.mlm_head(hidden)

        return logits

    def entity_embedding(self, entity_tokens):

        segment_ids = torch.zeros_like(entity_tokens)

        pad_mask = self._padding_mask(entity_tokens)

        hidden = self.encoder(entity_tokens, segment_ids, padding_mask=pad_mask)

        return self._masked_mean(hidden, entity_tokens)

    def relation_score(self, h_tokens, r, t_tokens):

        h = self.entity_embedding(h_tokens)
        t = self.entity_embedding(t_tokens)

        r_vec = self.relation_embeddings(r)

        score = torch.norm(h + r_vec - t, dim=1)

        return score

    def type_prediction(self, entity_tokens):

        entity_embedding = self.entity_embedding(entity_tokens)

        logits = self.type_head(entity_embedding)

        return logits
