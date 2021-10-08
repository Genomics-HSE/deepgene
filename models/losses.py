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


def CTC_loss(y_pred, y_true):
    """
    y_pred -- (batch_size, seq_len, dim)
    Log_probs: Tensor of size  (T,N,C)
    :return:
    """
    batch_size = y_pred.shape[0]
    seq_len = y_pred.shape[1]
    y_pred = y_pred.permute(1, 0, 2)
    y_pred = F.softmax(y_pred, dim=-1)
    seq_lengths = torch.full(size=(batch_size,), fill_value=seq_len).long()
    loss = F.ctc_loss(
        log_probs=y_pred,
        targets=y_true,
        input_lengths=seq_lengths,
        target_lengths=seq_lengths,
        blank=None,
    )
    return loss
