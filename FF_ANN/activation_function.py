import numpy as np


class ReLU:
    """
    In order to make our network to learn non-linear decision boundaries, ReLU 
    was used as the activation function. ReLU class (activation function) 
    implements two methods: forward propagation and backpropagation.
    """

    def __init__(self):
        """
        ReLu does not have any learning parameter to be initialized.
        """
        pass

    def forward(self, input):
        """
        Apply elementwise ReLU to input data of shape [batch, input_units] 
        and returns output data with the same shape"""
        relu_forward = np.maximum(0, input)
        return relu_forward

    def backward(self, input, grad_output):
        """
        Gradient of loss with respect to ReLU input
        """
        relu_grad = input > 0
        return grad_output * relu_grad
