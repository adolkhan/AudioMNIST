# AudioMNIST

Hi! In this repository you can find a character level model that transcribes recorded audious that contains a number in it. 

To train a model simply run `python train.py`

To inference a model simply run `python inference.py --output-file $OUTPUT_FILE`

The inferenced csv will be stored in data folder 

All of the data should be located in data folder:

![alt text](https://github.com/adolkhan/AudioMNIST/blob/main/image.png)


## Model description
To train a model i converted all audios into a mel spectrograms. 
Trained a model that consisted of 2 Convolutions with batch norms and activation + 1 biLSTM.
As a loss function used CTCLoss. Used several types of augmentations, the best ones that I decided to keep are GaussianNoise and Pitch Shifter.

# Checkpoints:
best model after 20 epochs - ![a link](https://github.com/adolkhan/AudioMNIST/blob/main/checkpoints/checkpoint_epoch_20.pth)
best model after 100 epochs - ![a link](https://github.com/adolkhan/AudioMNIST/blob/main/checkpoints/checkpoint_epoch_100.pth)
