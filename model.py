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
    def __init__(self, num_activities=22, input_dim=17*2, num_frames=32, embedding_size=256, num_heads=8, hidden_size=512, num_layers=6, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.num_activities = num_activities
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.embedding_size = embedding_size
        
        self.positional_encoding = nn.Parameter(torch.randn(num_frames, embedding_size))
        self.keypoints_embedding = nn.Linear(input_dim, embedding_size)
        self.frames_embedding = nn.Linear(num_frames, embedding_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embedding_size * num_frames * input_dim, num_activities)
        
    def forward(self, x):
        keypoints_embedded = self.keypoints_embedding(x)
        frames_embedded = self.frames_embedding(torch.arange(self.num_frames).unsqueeze(0).repeat(x.size(0), 1))
        embedded = keypoints_embedded + self.positional_encoding.unsqueeze(0)
        embedded += frames_embedded.unsqueeze(1)
        
        embedded = embedded.permute(1, 0, 2)  # (num_frames, batch_size, embedding_size)
        output = self.transformer_encoder(embedded)
        output = output.permute(1, 0, 2)  # (batch_size, num_frames, embedding_size)
        
        # Perform global pooling or aggregation
        output = torch.flatten(output, start_dim=1)
        
        output = self.fc(output)
        
        return output
