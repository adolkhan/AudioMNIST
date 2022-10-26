import torch.nn.functional as F
from torch import nn


class biLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_vocabs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True, bias=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, num_vocabs)

    def forward(self, input_mel):
        output, _ = self.lstm(input_mel.transpose(-1, -2))
        output = self.fc(output)
        output = F.log_softmax(output, dim=-1)
        return output.transpose(0, 1)
