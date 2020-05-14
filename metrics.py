import torch

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


def binary_accuracy(output1, output2, target):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    distances = (output2 - output1).pow(2).sum(1)
    rounded_preds = torch.round(torch.sigmoid(distances))
    correct = (rounded_preds == target).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc