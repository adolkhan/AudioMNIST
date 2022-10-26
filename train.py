import argparse

from trainer import Trainer
from utils import set_deterministic


def main(parser):
    args = parser.parse_args()
    set_deterministic()

    trainer = Trainer(
        args.model_path,
        args.epochs,
        args.train_dir,
    )

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-path", type=str,
                        default="/Users/adilkhansarsen/Documents/work/AudioMNIST/checkpoints/checkpoint.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-dir", type=str, default="data/train.csv")
    main(parser)
