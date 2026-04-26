import torch.nn as nn


class TypeHead(nn.Module):

    def __init__(self, hidden_dim, num_types):

        super().__init__()

        self.linear = nn.Linear(hidden_dim, num_types)

    def forward(self, x):

        return self.linear(x)