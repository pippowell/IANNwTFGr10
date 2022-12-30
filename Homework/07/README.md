# CNN + LSTM
Through the CNN layers our model identifies numbers in 28x28 images from the MNIST dataset. Then the numbers in sequence will be fed into the LSTM layer in order to be computed as such: if the sequence is [4,1,0,7], then after the computation it is [+4=4, 4-1=3, 3+0=3, 3-7=-4] 

## The order of running the files
1. [dataset.py](https://github.com/pippowell/IANNwTFGr10/blob/main/Homework/07/dataset.py)
: We define a method to preprocess the mnist dataset. The most important feature of the preprocessing is the double batching (for sequence_length and for batch)

2. [model.py](https://github.com/pippowell/IANNwTFGr10/blob/main/Homework/07/model.py)
: We first define a customized LSTM cell, which we will next implement in the main model combining it with CNN.
In the main model, we will feed the dataset through the CNN layers with some optimization (e.g. batch normalization), then through LSTM, and then finally through a Dense layer without an activation function. 

3. [training.py](https://github.com/pippowell/IANNwTFGr10/blob/main/Homework/07/training.py)
: We use compile and fit methods to train the model for default 15 epochs and with 1e-2 learning rate. Corresponding plots are saved in the folder [plots](https://github.com/pippowell/IANNwTFGr10/tree/main/Homework/07/plots) in the same directory as this README.md.

4. [load_model.py](https://github.com/pippowell/IANNwTFGr10/blob/main/Homework/07/load_model.py)
: Script to load the trained model (my_model07). Model was trained for 15 Epochs and 0.01 as learning rate.
