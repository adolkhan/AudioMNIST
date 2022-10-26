# import os.path
# import os
# import random
# from typing import List, Tuple
#
#
# import albumentations as A
# import numpy as np
# import torch, torchaudio
# import torch.nn.functional as F
# from torch import nn, optim
# from torch.utils.data import Dataset, DataLoader, Subset
# from torch.nn import CTCLoss
# import torchvision
#
# from augmentations import (
#     GaussianNoise,
#     TimeStreching,
#     PitchShifting,
#     VolumeChange,
#     AddNoise,
# )
# num_epcohs = 10
# PATH = "/Users/adilkhansarsen/Documents/work/AudioMNIST/checkpoints/model_weights_2.pth"
#
#
# class AudioDataLoader(Dataset):
#     def __init__(self, path_to_csv: str, transform=None):
#         data = self._parse_csv(path_to_csv)
#         self.root_folder = os.path.dirname(path_to_csv)
#         self.paths = data["path"]
#         self.genders = data["gender"]
#         self.labels = data["number\n"]
#         self.transform = transform
#         self.male_voice_paths, self.male_voice_labels = self._get_paths_and_labels("male")
#         self.female_voice_paths, self.female_voice_labels = self._get_paths_and_labels("female")
#         self.ratio = self._get_gender_ratio()
#
#     def __getitem__(self, index):
#         path_to_wav, label = self._get_sampled_item(index)
#         wav, sr = torchaudio.load(path_to_wav)
#         # wav = self.transform(wav.numpy())
#         label = torch.tensor([int(char) for char in label] + [11])
#         return wav, label
#
#     def __len__(self):
#         return len(self.paths)
#
#     def _parse_csv(self, path_to_csv):
#         data = {}
#         stream = self._open_stream(path_to_csv)
#         names = stream.readline().split(",")
#         csv_content = stream.read()
#         csv_content = csv_content.split("\n")
#         csv_content = list(map(lambda x: x.split(","), csv_content))
#         csv_content = list(filter(lambda x: len(x) == 3 and x[2] != "", csv_content))
#         csv_content = list(zip(*csv_content))
#
#         for i, name in enumerate(names):
#             data[name] = list(csv_content[i])
#         return data
#
#     def _get_paths_and_labels(self, gender):
#         male_voice_paths = []
#         male_voice_labels = []
#
#         for idx, value in enumerate(self.genders):
#             if value == gender:
#                 male_voice_paths.append(self.paths[idx])
#                 male_voice_labels.append(self.labels[idx])
#         return male_voice_paths, male_voice_labels
#
#     def _get_gender_ratio(self):
#         female_data_len = len(self.female_voice_labels)
#         male_data_len = len(self.male_voice_labels)
#         ratio = male_data_len/(male_data_len+female_data_len)
#         return ratio
#
#     def _get_sampled_item(self, index):
#         if random.random() < self.ratio:
#             index = index % len(self.female_voice_paths)
#             path_to_wav = os.path.join(self.root_folder, self.female_voice_paths[index])
#             label = self.female_voice_labels[index].split(".")[0]
#         else:
#             index = index % len(self.male_voice_paths)
#             path_to_wav = os.path.join(self.root_folder, self.male_voice_paths[index])
#             label = self.male_voice_labels[index].split(".")[0]
#         return path_to_wav, label
#
#     @staticmethod
#     def _open_stream(csv_file):
#         return open(csv_file, "r")
#
#
#
#
#
#
#
#
#
# train_transform = torchvision.transforms.Compose(
#     [
#         GaussianNoise(),
#         TimeStreching(),
#         PitchShifting(),
#         VolumeChange(),
#         # AddNoise(),
#     ]
# )
# dataset = AudioDataLoader("data/train.csv", transform=train_transform)
# train_dataset, validation_dataset = train_val_splitter(dataset)
#
#
#
#
#
# model = Net(input_dim=64, hidden_size=128, num_vocabs=13)
# featurizer = Featurizer()
# optimizer = optim.AdamW(model.parameters(), lr=1e-3)
#
# ctc_loss = CTCLoss(blank=10)
#
# train_loss = []
#
#
#
