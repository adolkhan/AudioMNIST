import torch.nn.functional as F
from torch import nn


class CNNbiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_vocabs):
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding="same")
        self.cnn2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding="same")
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, batch_first=True, bias=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, num_vocabs)

    def forward(self, input_mel):
        output = self.cnn1(input_mel)
        output = self.cnn2(output)
        output, _ = self.lstm(output.transpose(-1, -2))
        output = self.fc(output)
        output = F.log_softmax(output, dim=-1)
        return output.transpose(0, 1)
