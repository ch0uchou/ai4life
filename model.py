import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_num, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(32)

    def forward(self, inputs):
        x = self.bn(inputs)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(32)

    def forward(self, inputs):
        x = self.bn(inputs)
        embedded = self.embedding(x)
        
        transformer_out = self.transformer_encoder(embedded)
        out = self.fc(transformer_out[:, -1, :])
        
        return out
