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

        # self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3)
        # self.relu = nn.ReLU()
        # self.lstm1 = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=layer_num, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=layer_num, batch_first=True)
        # self.fc = nn.Linear(hidden_dim, 22)

    def forward(self, inputs):
        x = self.bn(inputs)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
        # x = inputs.permute(0, 2, 1)  # Reshape to (batch_size, 34, 32) for 1D conv
        # x = self.conv1d(x)
        # x = self.relu(x)

        # # Reshape back to (batch_size, 32, 64) for LSTM
        # x = x.permute(0, 2, 1)
        # # LSTM expects input of shape (batch_size, seq_len, input_size)

        # # First LSTM layer
        # lstm_out1, _ = self.lstm1(x)

        # # Second LSTM layer
        # lstm_out2, _ = self.lstm2(lstm_out1)

        # # Get output from the last time step
        # lstm_out = lstm_out2[:, -1, :]

        # # Fully connected layer
        # output = self.fc(lstm_out)
        # return output
