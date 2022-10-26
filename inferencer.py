import os.path

import torch
import torchaudio.functional
from torchaudio.models.decoder import ctc_decoder
from models.biLSTM import biLSTM
from models.maskedCNNbiLSTM import MaskedCNNbiLSTM
from models.CNNbiLSTM import CNNbiLSTM
from utils import Featurizer
from functools import reduce
import glob

from tokens import TOKENS


class Inferencer:
    LM_WEIGHT = 3.23
    WORD_SCORE = -0.26

    def __init__(self, model_path, tokens=TOKENS, outfile="data/test_predictions.csv"):
        self.featurizer = Featurizer()
        self.outfile = outfile
        self.model = MaskedCNNbiLSTM(input_dim=64, hidden_size=128, num_vocabs=13)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.beam_search_decoder = ctc_decoder(
            lexicon=None,
            tokens=tokens,
            nbest=3,
            beam_size=5,
            lm_weight=self.LM_WEIGHT,
            word_score=self.WORD_SCORE,
        )

    def run(self, test_dir="data/test-example/"):
        self.model.eval()

        with open(self.outfile, "w") as f:
            for file_path in glob.glob(os.path.join(test_dir, "*")):
                file, sr = torchaudio.load(file_path)
                mel, mel_length = self.featurizer(file, file.shape[-1])
                output, mel_length = self.model(mel, torch.tensor([mel_length]).long())
                # output = output.transpose(1, 0)
                out = self.beam_search_decoder(output)
                out = list(filter(lambda x: x != 11, out[0][0].tokens.tolist()))
                try:
                    out_str = reduce(lambda x, y: str(x)+str(y), out)
                except:
                    out_str = "None"

                f.write(file_path + "," + str(out_str)+"\n")
