import numpy as np
from loss_gradloss import softmax_crossentropy, grad_softmax_crossentropy
from tqdm import trange


def forward(network, X):
    """
    Return a list of activations for each layer by applying them sequentially. 
    """
    activations = []
    input = X

    for layer in network:
        activations.append(layer.forward(input))
        input = activations[-1]

    return activations


def train(network, X, y):
    """
    Train the network on a given batch of X and y.
    First forward gives all layer activations. Then layer.backward goes from 
    last to first layer.
    """

    # Get the layer activations
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
    logits = layer_activations[-1]

    # Compute the loss and the initial gradient
    loss = softmax_crossentropy(logits, y)
    loss_grad = grad_softmax_crossentropy(logits, y)

    # Backpropagation
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        loss_grad = layer.backward(layer_inputs[layer_index],
                                   loss_grad)  # grad with respect to input and update weights

    return np.mean(loss)


def predict(network, X):
    """
    Return network predictions that are indices of largest Logit probability
    """
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)


def iterate_minibatches(inputs, targets, batchsize):
    indices = np.random.permutation(len(inputs))  # shuffle orders
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]
