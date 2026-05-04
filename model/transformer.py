import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, heads):
        super().__init__()

        assert hidden_dim % heads == 0

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # reshape into heads
        Q = Q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # attention scores
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn = torch.softmax(scores, dim=-1)

        out = attn @ V

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, heads, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadSelfAttention(hidden_dim, heads)
        self.ff = FeedForward(hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention block
        attn_out = self.attn(x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feedforward block
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim=256, heads=8, layers=6):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, heads)
            for _ in range(layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

