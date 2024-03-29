import torch
from torch.nn import Transformer

import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num):
        super(TransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.transformer = nn.Transformer(d_model=input_dim, nhead=layer_num, num_encoder_layers=layer_num)
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(32)

    def forward(self, inputs):
        x = self.bn(inputs)
        transformer_out = self.transformer(x)
        out = self.fc(transformer_out[:, -1, :])
        return out
