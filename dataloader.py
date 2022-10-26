from torch.utils.data import DataLoader

from datasets import AudioDataset
from utils import Collator, train_val_splitter


def create_dataloader(path, transform, batch_size=16):
    dataset = AudioDataset(path, transform=transform)
    train_dataset, validation_dataset = train_val_splitter(dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Collator(),
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Collator(),
    )
    return train_dataloader, validation_dataloader
