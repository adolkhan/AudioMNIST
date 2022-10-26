# AudioMNIST

Hi! In this repository you can find a character level model that transcribes recorded audious that contains a number in it. 

To train a model simply run `python train.py`

To inference a model simply run `python inference.py --output-file $OUTPUT_FILE`

The inferenced csv will be stored in data folder 

All of the data should be located in data folder:

![alt text](https://github.com/adolkhan/AudioMNIST/blob/main/image.png)


To train a model i converted all audios into a mel spectrograms. 
Trained a model that consisted of 2 Convolutions with batch norms and activation + 1 biLSTM.
As a loss function used CTCLoss. Used several types of augmentations, the best ones that I decided to keep are GaussianNoise and Pitch Shifter.
