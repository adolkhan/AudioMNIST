import os
import random
from typing import List, Tuple

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.utils.data import Subset


class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()

        self.featurizer = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=64,
            center=True
        )

    def forward(self, wav, length=None):
        mel_spectrogram = self.featurizer(wav)
        mel_spectrogram = mel_spectrogram.clamp(min=1e-5).log()

        if length is not None:
            length = (length - self.featurizer.win_length) // self.featurizer.hop_length
            # We add `4` because in MelSpectrogram center==True
            length += 1 + 4

            return mel_spectrogram, length

        return mel_spectrogram


class Collator:
    def __call__(self, batch: List[Tuple[torch.Tensor, int]]):
        wav_lengths = []
        label_lengths = []
        wavs, labels = zip(*batch)

        for wav in wavs:
            try:
                wav_lengths.append(wav.size(-1))
            except:
                pass

        for label in labels:
            label_lengths.append(len(label))


        max_wav_length = max(wav_lengths)
        max_label_length = max(label_lengths)

        batch_wavs = torch.cat(
            list(
                map(lambda x: F.pad(
                    x,
                    pad=(0, max_wav_length-x.size(-1)),
                    mode="constant",
                    value=0
                ), wavs)
            )
        )

        batch_labels = torch.cat(
            list(
                map(lambda x: F.pad(
                    x.unsqueeze(0),
                    pad=(0, max_label_length-x.size(-1)),
                    mode="constant",
                    value=99
                ), labels)
            )
        )

        batch_labels = batch_labels.long()
        wav_lengths = torch.tensor(wav_lengths).long()
        label_lengths = torch.tensor(label_lengths).long()

        return {
            "wav": batch_wavs,
            "label": batch_labels,
            "wav_lengths": wav_lengths,
            "label_lengths": label_lengths
        }


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_val_splitter(dataset, ratio=0.9):
    train_size = int(len(dataset)*ratio)

    indexes = torch.randperm(len(dataset))
    train_indexes = indexes[:train_size]
    validation_indexes = indexes[train_size:]

    train_dataset = Subset(dataset, train_indexes)
    validation_dataset = Subset(dataset, validation_indexes)
    return train_dataset, validation_dataset


def set_deterministic(seed=404, determenistic=True):
    if determenistic:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.determenistic = determenistic