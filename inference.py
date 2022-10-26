import argparse

from inferencer import Inferencer
from utils import set_deterministic


def main(parser):
    args = parser.parse_args()
    set_deterministic()

    trainer = Inferencer(
        model_path=args.model_path,
        outfile=args.output_file,
    )

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="/Users/adilkhansarsen/Documents/work/AudioMNIST/checkpoints/checkpoint.pth")
    parser.add_argument("--output-file", type=str, default="data/test_predictions.csv")
    main(parser)
