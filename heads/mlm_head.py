import torch.nn as nn

class MLMHead(nn.Module):

    def __init__(self, hidden_dim, vocab_size):

        super().__init__()

        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):

        return self.linear(x)