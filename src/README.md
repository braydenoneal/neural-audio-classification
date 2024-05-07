# Neural Audio Classification

The goal of this project is to classify audio using a neural network.

## Usage

1. Download the [Audio MNIST](https://github.com/soerenab/AudioMNIST) repository and copy the contents of the `data`
   directory into the `sounds` directory of this project.
2. Run the [spectrogram.py](spectrogram.py) file to generate the audio images that the neural network will train on.
3. Run the [neural_network.py](neural_network.py) to train and export the neural network.
4. Run the [record.py](record.py) to record audio for testing the model.
5. Run the [predict.py](predict.py) file to test the model on the recorded audio.
