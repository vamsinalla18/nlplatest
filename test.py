import torch
from model.bert_encoder import BERTEncoder
from heads.mlm_head import MLMHead

vocab_size = 30000

encoder = BERTEncoder(vocab_size)
mlm = MLMHead(256, vocab_size)

input_ids = torch.randint(0, vocab_size, (2, 32))
segment_ids = torch.zeros((2,32)).long()

hidden = encoder(input_ids, segment_ids)

output = mlm(hidden)

print("Hidden shape:", hidden.shape)
print("MLM output shape:", output.shape)