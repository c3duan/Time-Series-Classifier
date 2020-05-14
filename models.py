import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, encode_dim, hidden_dim, output_dim, num_layers, dropout):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(
            encode_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def encode(self, inputs):
        inputs = inputs.T.view(*inputs.T.shape, 1)
        packed_output, (hidden, cell) = self.lstm(inputs)
        hidden = hidden[-1, :, :].squeeze(0)
        return hidden

    def forward(self, inputs):
        inputs = inputs.T.view(*inputs.T.shape ,1)
        packed_outputs, (hidden, cell) = self.lstm(inputs)
        hidden = self.dropout(hidden[-1,:,:].squeeze(0))
        outputs = self.fc(hidden)
        return outputs


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.main = nn.Sequential(  # W_in = 10 * 1
            nn.Conv1d(1, 10, 3, stride=1, padding=1, dilation=1, bias=True),  # W_out = 10 * 10
            nn.ReLU(inplace=True),
            nn.Conv1d(10, 5, 5, stride=2, padding=1, bias=True),  # W_out = 4 * 5
            nn.BatchNorm1d(5),  # Suppose to be the number of channels
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2) # W_out = 2 * 5
        )
        self.fc = nn.Sequential(
            nn.Linear(10, 64),  # The choice of 64 is completely random
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 1)
        )

    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

    def encode(self, inputs):
        batch_size, input_dim = inputs.shape
        inputs = inputs.view(batch_size, 1, input_dim)
        x = self.main(inputs)
        x = x.view(batch_size, self.num_flat_features(x))
        return x

    def forward(self, inputs):
        batch_size, input_dim = inputs.shape
        inputs = inputs.view(batch_size, 1, input_dim)
        x = self.main(inputs)
        x = x.view(batch_size, self.num_flat_features(x))
        return self.fc(x)


class RNNSiameseNet(nn.Module):
    def __init__(self, encode_dim, hidden_dim, output_dim, num_layers, dropout):
        super(RNNSiameseNet, self).__init__()
        self.lstm = nn.LSTM(
            encode_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward_once(self, inputs):
        inputs = inputs.T.view(*inputs.T.shape, 1)
        packed_outputs, (hidden, cell) = self.lstm(inputs)
        hidden = self.dropout(hidden[-1, :, :].squeeze(0))
        outputs = self.fc(hidden)
        return outputs

    def forward(self, item1, item2):
        output1 = self.forward_once(item1)
        output2 = self.forward_once(item2)
        return output1, output2


class CNNSiameseNet(nn.Module):
    def __init__(self):
        super(CNNSiameseNet, self).__init__()
        self.main = nn.Sequential(  # W_in = 10 * 1
            nn.Conv1d(1, 10, 3, stride=1, padding=1, dilation=1, bias=True),  # W_out = 10 * 10
            nn.ReLU(inplace=True),
            nn.Conv1d(10, 5, 5, stride=2, padding=1, bias=True),  # W_out = 4 * 5
            nn.BatchNorm1d(5),  # Suppose to be the number of channels
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # W_out = 2 * 5
        )
        self.fc = nn.Sequential(
            nn.Linear(10, 64),  # The choice of 64 is completely random
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 1)
        )

    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

    def forward_once(self, inputs):
        batch_size, input_dim = inputs.shape
        inputs = inputs.view(batch_size, 1, input_dim)
        x = self.main(inputs)
        x = x.view(batch_size, self.num_flat_features(x))
        return self.fc(x)

    def forward(self, item1, item2):
        output1 = self.forward_once(item1)
        output2 = self.forward_once(item2)
        return output1, output2
