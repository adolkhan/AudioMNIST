import random

import numpy as np
import torch
from torch import distributions
import torchaudio
import librosa


class GaussianNoise:
    def __init__(self, loc=0, validate_args=None, p=0.5):
        self.scale = random.random()/10
        self.validate_args = validate_args
        self.loc = loc
        self.p = p

    def __call__(self, wav):
        scale = random.random() / 10
        noiser = distributions.Normal(loc=self.loc, scale=scale, validate_args=self.validate_args)
        if random.random() < self.p:
            if isinstance(wav, np.ndarray):
                wav = torch.tensor(wav)
            return wav + noiser.sample(wav.size())
        if isinstance(wav, np.ndarray):
            return torch.from_numpy(wav)
        return wav

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class TimeStreching:
    def __init__(self, p=1):
        super(TimeStreching, self).__init__()
        self.p = p

    def __call__(self, wav):
        if random.random() < self.p:
            rate = 2 * random.random()
            if not isinstance(wav, np.ndarray):
                wav = wav.numpy()
            augumented_wav = librosa.effects.time_stretch(wav.squeeze(), rate)
            return torch.from_numpy(augumented_wav).unsqueeze(0)
        if isinstance(wav, np.ndarray):
            return torch.from_numpy(wav)
        return wav

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class PitchShifting:
    def __init__(self, sr=24000, p=0.5):
        super(PitchShifting, self).__init__()
        self.sr = sr
        self.p = p

    def __call__(self, wav):
        if random.random() < self.p:
            if not isinstance(wav, np.ndarray):
                wav = wav.numpy()
            level = random.randint(-5, 5)
            if level == 0:
                level = 0.1
            augumented_wav = librosa.effects.pitch_shift(wav.squeeze(), sr=self.sr, n_steps=level)
            return torch.from_numpy(augumented_wav).unsqueeze(0)
        if isinstance(wav, np.ndarray):
            return torch.from_numpy()
        return wav

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class VolumeChange:
    def __init__(self, p=0.5):
        super(VolumeChange, self).__init__()
        self.p = p

    def __call__(self, wav):
        if random.random() < self.p:
            gain = random.random()/10
            voler = torchaudio.transforms.Vol(gain=gain, gain_type='amplitude')
            if isinstance(wav, np.ndarray):
                wav = torch.tensor(wav)
            augumented_wav = voler(wav)
            return augumented_wav
        if isinstance(wav, np.ndarray):
            return torch.from_numpy(wav)
        return wav

    def __repr__(self):
        return f"{self.__class__.__name__}()"
