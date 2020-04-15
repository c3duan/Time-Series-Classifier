import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, encode_dim, hidden_dim, output_dim, num_layers, dropout):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(
            encode_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        data = data.reshape(1, *data.shape)
        packed_output, (hidden, cell) = self.lstm(data)
        hidden = self.dropout(hidden[-1,:,:])
        output = self.fc(hidden)

        return output
