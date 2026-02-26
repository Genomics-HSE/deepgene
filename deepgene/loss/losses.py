import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import numpy as np

patch_typeguard()


limits = [
0.0,
2900,
6000,
9500,
15000,
18000,
30000,
42000
]


def from_probability_to_N(priors, limits=limits):
    l = 1
    tau = torch.tensor(np.array(limits[1:]) - np.array(limits[:-1]))
    N = []
    for j in range(len(tau)):
        N.append(
            torch.round(tau[j]/(-2*torch.log(1 - (priors[j]/l))))
        )
        l *= torch.exp(-tau[j]/(2*N[j]))
    return N

def NeLoss(n_class: int,
           y_pred, # : TensorType["batch", "genome_length", "hidden_size"],
           y_true #: TensorType["batch", "genome_length"],
           ):
    y_pred = F.log_softmax(y_pred, dim=-1)
    y_true = y_true.long()
    y_true = F.one_hot(y_true, n_class).float()

    # Превратить их в распределение
    d_pred = y_pred.sum(axis=0)/len(y_true)
    d_true = y_true.sum(axis=0)/len(y_true)
    # Распределение в Ne
    N_pred = from_probability_to_N(d_pred)
    N_true = from_probability_to_N(d_true)
    # Считаем абсолюты отклонений Ne в каждом интервале

    
    loss = [abs(n - m) for (n,m) in zip(N_pred, N_true)]
    
    return loss


@typechecked
def L1Loss(n_class: int,
           y_pred: TensorType["batch", "genome_length", "hidden_size"],
           y_true: TensorType["batch", "genome_length"],
           ):
    y_pred = F.log_softmax(y_pred, dim=-1)
    y_true = y_true.long()
    y_true = F.one_hot(y_true, n_class).float()
    loss = F.l1_loss(
        y_pred, y_true
        , reduction='mean'
    )
    return loss

@typechecked
def KLDivLoss(n_class: int,
              y_pred: TensorType["batch", "genome_length", "hidden_size"],
              y_true: TensorType["batch", "genome_length"]):
    assert len(y_true.shape) == 2
    y_pred = F.log_softmax(y_pred, dim=-1)
    y_true = y_true.long()
    y_true = F.one_hot(y_true, n_class).float()
    loss = F.kl_div(y_pred, y_true, reduction='batchmean')
    return loss


@typechecked
def NllLoss(
             y_pred: TensorType["batch", "genome_length", "hidden_size"],
             y_true: TensorType["batch", "genome_length"]
            ):
    # nll_loss
    y_pred = F.log_softmax(y_pred, dim=-1)
    y_pred = y_pred.permute(0, 2, 1)
    y_true = y_true.long()
    loss = F.nll_loss(y_pred, y_true)
    return loss 


@typechecked
def HuberLoss(n_class: int,
           y_pred: TensorType["batch", "genome_length", "hidden_size"],
           y_true: TensorType["batch", "genome_length"],
           ):
    y_pred = F.log_softmax(y_pred, dim=-1)
    y_true = y_true.long()
    y_true = F.one_hot(y_true, n_class).float()
    loss = F.huber_loss(
        y_pred, y_true
        , reduction='mean'
        , delta=1.0
    )
    return loss


@typechecked
def CrossEntropyLoss(y_pred: TensorType["batch", "genome_length", "hidden_size"],
                     y_true: TensorType["batch", "genome_length"],
                     weight=None):
    # y_pred = F.log_softmax(y_pred, dim=-1)
    y_pred = y_pred.permute(0, 2, 1)
    output = F.cross_entropy(y_pred, y_true.long(), weight=weight)
    return output


@typechecked
def EMD_squared_loss(n_class: int,
                     y_pred: TensorType["batch", "genome_length", "hidden_size"],
                     y_true: TensorType["batch", "genome_length"]):
    y_pred = F.softmax(y_pred, dim=-1)
    y_true = F.one_hot(y_true.long(), n_class).float()
    output = torch.mean(
        torch.sum(
            torch.square(
                torch.cumsum(y_pred, dim=-1) - torch.cumsum(y_true, dim=-1)
            ), dim=-1)
    )
    return output

@typechecked
def BCEWithLogitsLoss(n_class: int,
             y_pred: TensorType["batch", "genome_length", "hidden_size"],
             y_true: TensorType["batch", "genome_length"]):
    y_pred_ce = y_pred.permute(0, 2, 1)
    loss_ce = F.cross_entropy(y_pred_ce, y_true.long())
    y_pred = F.softmax(y_pred, dim=-1)
    y_true = F.one_hot(y_true.long(), n_class).float()
    loss_emd = torch.mean(torch.square(torch.cumsum(y_pred, dim=-1) - torch.cumsum(y_true, dim=-1)))
    return loss_ce + 2*loss_emd

@typechecked
def CEandEMD(n_class: int,
             y_pred: TensorType["batch", "genome_length", "hidden_size"],
             y_true: TensorType["batch", "genome_length"]):
    y_pred_ce = y_pred.permute(0, 2, 1)
    loss_ce = F.cross_entropy(y_pred_ce, y_true.long())
    y_pred = F.softmax(y_pred, dim=-1)
    y_true = F.one_hot(y_true.long(), n_class).float()
    loss_emd = torch.mean(torch.square(torch.cumsum(y_pred, dim=-1) - torch.cumsum(y_true, dim=-1)))
    return loss_ce + 2*loss_emd


@typechecked
def CTC_loss(y_pred: TensorType["batch", "genome_length", "hidden_size"],
             y_true: TensorType["batch", "genome_length"]):
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


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
    
    def forward(self, input, target):
        input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def MYLOSS(n_class, device):
    D = torch.full(size=(n_class, n_class), fill_value=0, device=device).float()
    
    for i in range(n_class):
        for j in range(n_class):
            if i == j:
                continue
            D[i][j] = torch.exp(torch.tensor((abs(i - j))))
    
    D = F.softmax(D, dim=-1)
    print(D)
    
    def emd_loss(y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=-1)
        
        emd2_loss = torch.square(y_pred) * F.embedding(input=y_true, weight=D)
        emd2_loss = torch.sum(emd2_loss, dim=-1)
        emd2_loss = torch.mean(emd2_loss)
        
        cross_loss = F.nll_loss(y_pred.permute(0, 2, 1), y_true)
        
        return cross_loss + emd2_loss
    
    return emd_loss
