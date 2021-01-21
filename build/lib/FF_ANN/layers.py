import numpy as np


class Dense:
    """
    Hidden layer implements two methods: forward propagation and 
    backpropagation
    """

    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        Xavier initialization of the weights (to converge faster)
        """
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (input_units + output_units)),
                                        size=(input_units, output_units))
        self.biases = np.zeros(output_units)

    def forward(self, input):
        """
        f(x) = <W*x> + b
        input shape: [batch, input_units]
        output shape: [batch, output units]
        b shape [batch, output units]
        """
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        """
        Backpropagation through the layer. Loss gradients with respect to input is:
        d loss / d x = (d loss / d dense) * (d dense / d x) = input * (d dense / d x)
        where d dense/ d x = weights transposed
        """
        # gradient
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]

        # stochastic gradient descent
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input
