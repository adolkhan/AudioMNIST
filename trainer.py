from itertools import islice
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch
import torchvision
import wandb
from torch import optim
from torch.nn import CTCLoss

from augmentations import (
    GaussianNoise,
    TimeStreching,
    PitchShifting,
    VolumeChange,
)
from utils import AverageMeter, Featurizer
from dataloader import create_dataloader
from models.biLSTM import biLSTM
from models.CNNbiLSTM import CNNbiLSTM
from models.maskedCNNbiLSTM import MaskedCNNbiLSTM


class Trainer:
    def __init__(self, model_path, epochs=20, train_dir="data/train.csv", batch_size=16, use_wandb=False):
        self.featurizer = Featurizer()
        transforms = torchvision.transforms.Compose(
            [
                GaussianNoise(),
                PitchShifting(),
            ]
        )
        self.train_dataloader, self.validation_dataloader = \
            create_dataloader(train_dir, transform=transforms, batch_size=batch_size)
        self.model = MaskedCNNbiLSTM(input_dim=64, hidden_size=128, num_vocabs=13)
        self.featurizer = Featurizer()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.model_path = model_path
        self.ctc_loss = CTCLoss(blank=10)
        self.epochs = epochs
        self.storage = defaultdict(list)
        self.wandb = wandb.init(project="biLSTM") if use_wandb else False

    def run(self):
        min_loss = np.Inf
        for epoch in range(self.epochs):
            print("-"*20+f"epoch {epoch}"+"-"*20)
            train_loss_meter = AverageMeter()
            self.model.train()
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                wav = batch["wav"]
                label = batch["label"]
                wav_length = batch["wav_lengths"]
                target_lengths = batch["label_lengths"]

                mel, mel_length = self.featurizer(wav, wav_length)

                output, mel_length = self.model(mel, mel_length)
                loss = self.ctc_loss(log_probs=output.transpose(1, 0), targets=label, input_lengths=mel_length, target_lengths=target_lengths)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss_meter.update(loss.item())

            self.storage['train_loss'].append(train_loss_meter.avg)
            print(f"train loss: {train_loss_meter.avg}")
            if self.wandb:
                self.wandb.log({"train_loss": train_loss_meter.avg})

            self.model.eval()

            validation_loss_meter = AverageMeter()

            for i, batch in islice(enumerate(tqdm(self.validation_dataloader)), 1):
                wav = batch['wav']
                wav_length = batch['wav_lengths']
                label = batch['label']
                target_lengths = batch["label_lengths"]

                with torch.no_grad():
                    mel, mel_length = self.featurizer(wav, wav_length)
                    output, mel_length = self.model(mel, mel_length)

                    loss = self.ctc_loss(log_probs=output.transpose(1, 0), targets=label, input_lengths=mel_length, target_lengths=target_lengths)

                validation_loss_meter.update(loss.item())
            self.storage['validation_loss'].append(validation_loss_meter.avg)
            if self.wandb:
                self.wandb.log({"validation_loss": validation_loss_meter.avg})

            if validation_loss_meter.avg < min_loss:
                torch.save(self.model.state_dict(), self.model_path)
                min_loss = validation_loss_meter.avg

            print(f"validation_loss: {validation_loss_meter.avg}")
