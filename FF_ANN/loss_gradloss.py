import numpy as np


def softmax_crossentropy(logits, y_true):
    """Crossentropy from logits[batch, n_classes] and y_true"""
    a_correct = logits[np.arange(len(logits)), y_true]
    ce_loss = - a_correct + np.log(np.sum(np.exp(logits), axis=-1))
    return ce_loss


def grad_softmax_crossentropy(logits, y_true):
    """Crossentropy gradient from logits[batch, n_classes] and y_true"""
    ones_grad = np.zeros_like(logits)
    ones_grad[np.arange(len(logits)), y_true] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    grad_ce_loss = (- ones_grad + softmax) / logits.shape[0]
    return grad_ce_loss
