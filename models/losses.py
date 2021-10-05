import torch
import torch.nn.functional as F


def KLDivLoss(n_class, y_pred, y_true):
    assert len(y_true.shape) == 2
    y_pred = F.softmax(y_pred, dim=-1)
    y_true = F.one_hot(y_true, n_class).float()
    loss = F.kl_div(y_pred, y_true, reduction='batchmean')
    return loss


def CrossEntropyLoss(y_pred, y_true):
    y_pred = y_pred.permute(0, 2, 1)
    output = F.cross_entropy(y_pred, y_true)
    return output


def EMD_squared_loss(n_class, y_pred, y_true):
    y_pred = F.softmax(y_pred, dim=-1)
    y_true = F.one_hot(y_true, n_class).float()
    output = torch.mean(torch.square(torch.cumsum(y_pred, dim=-1) - torch.cumsum(y_true, dim=-1)))
    return output
