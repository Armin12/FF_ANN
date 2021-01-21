# Feedforward Neural Network Package

This is a simple neural network package built using numpy. No third party neural network library has been used (Tensorflow, Pytorch, Keras, etc.), except sklearn.datasets that was used to load iris dataset. This code contains an example that classifies iris dataset.

For this implementation we have considered sequential network model with dense hidden layers (Dense class) and ReLU activation functions (ReLU class). These classes include forward and backward methods for forward propagation and backpropagation. Feedforward step takes an input and generates output for making a prediction. Backpropagation is used for training by adjusting weights in the layer to minimize loss function.

Since our goal is to solve a classification problem with three classes, categorical cross-entropy with softmax has been used as the loss function and the optimizer is the stochastic gradient descent method. By writing cross-entropy as a function of softmax logits, loss will be an expression called Log-softmax which is numerically more stable, easier to calculate gradient, and faster to compute. We used this combination in our implementation in loss_gradloss.py. The file named network_training.py also includes functions related to the training.

Implemented neural network is used to classify iris data. In order not to use third party neural network library, data standardization is implemented in numpy. iris dataset contains 150 data, and we put 100 for the training set, 25 for the validation set, and 25 for the test set. The input to the network has 4 dimensions (which is the number of features in the iris dataset). The network is 10-20-3 which means three dense hidden layers with 10, 20, and 3 neurons from input toward the output. The network was trained in 50 epochs and with batch size of 16 data samples. Since, we have a balanced dataset, i.e. 50 samples for each class, accuracy metric was enough to measure the performance. So, we used accuracy. 

To use this package, you just need to run iris_classification.py.

