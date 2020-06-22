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


def metric_at_k(k, model, queries, dataloader, device):
    with torch.no_grad():
        correct_count = 0
        for item, label in zip(dataloader.dataset.data, dataloader.dataset.labels):
            item = item.repeat((len(queries), 1)).to(device)
            output1, output2 = model.forward_once(item), model.forward_once(queries)
            distances = (output2 - output1).pow(2).sum(1)
            ranking = distances.argsort()

            correct_count += float((ranking[:k] == label).nonzero().numel() > 0)

    return correct_count / len(dataloader.dataset.labels)