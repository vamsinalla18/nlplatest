import torch.nn as nn

class TransformerEncoder(nn.Module):

    def __init__(self, hidden_dim=256, heads=8, layers=6):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

    def forward(self, x, mask=None):

        return self.encoder(x, src_key_padding_mask=mask)