import torch.nn.functional as F


def KLDivLoss(n_class, y_pred, y_true):
    assert len(y_true.shape) == 2
    y_pred = F.softmax(y_pred, dim=-1)
    y_true = F.one_hot(y_true, n_class).float()
    loss = F.kl_div(y_pred, y_true, reduction='batchmean')
    return loss
