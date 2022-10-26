from typing import Tuple

import torch.nn.functional as F
import torch
from torch import nn
import math
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MaskConv(nn.Module):
    """
        Masking Convolutional Neural Network
        Refer to https://github.com/sooftware/KoSpeech/blob/jasper/kospeech/models/modules.py
        Copyright (c) 2020 Soohwan Kim
    """

    def __init__(
            self,
            sequential: nn.Sequential
    ) -> None:
        super(MaskConv, self).__init__()
        self.sequential = sequential

    def forward(
            self,
            inputs: Tensor,
            seq_lens: Tensor
    ) -> Tuple[Tensor, Tensor]:
        output = None

        for module in self.sequential:
            output = module(inputs)

            mask = torch.BoolTensor(output.size()).fill_(0)
            if output.is_cuda:
                mask = mask.cuda()

            seq_lens = self.get_seq_lens(module, seq_lens)

            for idx, seq_len in enumerate(seq_lens):
                seq_len = seq_len.item()

                if (mask[idx].size(2) - seq_len) > 0:
                    mask[idx].narrow(2, seq_len, mask[idx].size(2) - seq_len).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lens

    def get_seq_lens(
            self,
            module: nn.Module,
            seq_lens: Tensor
    ) -> Tensor:
        if isinstance(module, nn.Conv2d):
            seq_lens = seq_lens + (2 * module.padding[1]) - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lens = seq_lens.float() / float(module.stride[1])
            seq_lens = seq_lens.int() + 1

        if isinstance(module, nn.MaxPool2d):
            seq_lens >>= 1

        return seq_lens.int()


class MiniDeepSpeech(nn.Module):
    def __init__(self, input_dim, hidden_size, num_vocabs, num_layers=1, dropout=0.3, bidirectional=True):
        super().__init__()

        input_size = int(math.floor(input_dim + 2 * 10 - 31) / 3 + 1)
        input_size = int(math.floor(input_size + 2 * 5 - 16) / 3 + 1)
        input_size <<= 5
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, True, True, dropout, bidirectional)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(31, 11), stride=(3, 3), padding=(10, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(min_val=0, max_val=20, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(16, 11), stride=(3, 1), padding=(5, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(min_val=0, max_val=20, inplace=True)
        ))
        self.rnn_output_size = hidden_size << 1 if bidirectional else hidden_size
        self.fc = nn.Linear(self.rnn_output_size, num_vocabs)

    def forward(self,
                inputs: Tensor,
                input_lengths: Tensor
                ) -> Tuple[Tensor, Tensor]:
        inputs = inputs.unsqueeze(1)
        conv_output, output_lengths = self.conv(inputs, input_lengths)  # conv_output shape : (B, C, D, T)

        conv_output = conv_output.permute(0, 3, 1, 2)
        batch, seq_len, num_channels, hidden_size = conv_output.size()
        conv_output = conv_output.view(batch, seq_len, -1).contiguous()  # (B, T, C * D)

        conv_output = pack_padded_sequence(conv_output, output_lengths, batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(conv_output)
        rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)

        rnn_output_prob = self.fc(rnn_output)
        rnn_output_prob = F.log_softmax(rnn_output_prob, dim=-1)  # (B, T, num_vocabs)

        return rnn_output_prob, output_lengths