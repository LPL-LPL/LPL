from numpy.core.numeric import cross
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def regression_loss(x, y):
    # x, y are in shape (N, C)
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1)


def entropy(p):
    return Categorical(probs=p).entropy()


def entropy_loss(logits, reduction='mean'):

    losses = entropy(F.softmax(logits, dim=1))  # (N)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def reversed_cross_entropy(logits, labels, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    labels = torch.clamp(labels, min=1e-4, max=1.0)
    losses = -torch.sum(pred * torch.log(labels), dim=1)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def normalized_cross_entropy(logits, labels, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    log_logits = F.log_softmax(logits, dim=1)
    losses = - torch.sum(labels * log_logits, dim=1) / ( - torch.bmm(labels.unsqueeze(dim=2), log_logits.unsqueeze(dim=1)).sum(dim=(1,2)))

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def symmetric_cross_entropy(logits, labels, alpha, beta, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    ce = cross_entropy(logits, labels, reduction=reduction)
    rce = reversed_cross_entropy(logits, labels, reduction=reduction)
    return alpha * ce + beta * rce


def generalized_cross_entropy(logits, labels, rho=0.7, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    losses = torch.sum(labels * ((1.0 - torch.pow(pred, rho)) / rho), dim=1)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def normalized_generalized_cross_entropy(logits, labels, rho=0.7, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    pred_pow = torch.pow(pred, rho)
    losses = (1 - torch.sum(labels * pred_pow, dim=1)) / (C - torch.bmm(labels.unsqueeze(dim=2), pred_pow.unsqueeze(dim=1)).sum(dim=(1,2)))

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def mae_loss(logits, labels, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    pred = logits.softmax(dim=1)
    losses = torch.abs(pred - labels).sum(dim=1)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / N
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def mse_loss(logits, labels, reduction='none'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'
    pred = logits.softmax(dim=1)
    losses = torch.sum((pred - labels)**2, dim=1)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / N
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def active_passive_loss(logits, labels, alpha=10.0, beta=1.0, active='nce', passive='mae', rho = 0.7, reduction='none'):
    """
    ICML 2020 - Normalized Loss Functions for Deep Learning with Noisy Labels
    https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    
    a loss is deﬁned “Active” if it only optimizes at q(k=y|x)=1, otherwise, a loss is deﬁned as “Passive”

    :param logits: shape: (N, C)
    :param labels: shape: (N)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    if active == 'ce':
        active_loss = cross_entropy(logits, labels, reduction=reduction)
    elif active == 'nce':
        active_loss = normalized_cross_entropy(logits, labels, reduction=reduction)
    elif active == 'gce':
        active_loss = generalized_cross_entropy(logits, labels, rho=rho, reduction=reduction)
    elif active == 'ngce':
        active_loss = normalized_generalized_cross_entropy(logits, labels, rho=rho, reduction=reduction)
    else:
        raise AssertionError(f'active loss: {active} is not supported yet')

    if passive == 'mae':
        passive_loss = mae_loss(logits, labels, reduction=reduction)
    elif passive == 'mse':
        passive_loss = mse_loss(logits, labels, reduction=reduction)
    elif passive == 'rce':
        passive_loss = reversed_cross_entropy(logits, labels, reduction=reduction)
    else:
        raise AssertionError(f'passive loss: {passive} is not supported yet')

    return  alpha * active_loss + beta * passive_loss
    

def label_smoothing_cross_entropy(logits, labels, epsilon=0.1, reduction='none'):
    N = logits.size(0)
    C = logits.size(1)
    smoothed_label = torch.full(size=(N, C), fill_value=epsilon / (C - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    if logits.is_cuda:
        smoothed_label = smoothed_label.cuda()
        # print(smoothed_label)
    return cross_entropy(logits, smoothed_label, reduction)


class SmoothingLabelCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self._epsilon = epsilon
        self._reduction = reduction

    def forward(self, logits, labels):
        return label_smoothing_cross_entropy(logits, labels, self._epsilon, self._reduction)


class ScatteredCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self._reduction = reduction

    def forward(self, logits, labels):
        return cross_entropy(logits, labels, self._reduction)


def get_dataset_distributed(logits, labels, indices, eps=0.1): 

    ce_losses = label_smoothing_cross_entropy(logits, labels, epsilon=eps, reduction='none')
    ind_loss_sorted = torch.argsort(ce_losses.data)
    num_remember = torch.nonzero(ce_losses < ce_losses.mean()).shape[0]

    ind_clean = ind_loss_sorted[:num_remember]
    ind_forget = ind_loss_sorted[num_remember:]
    logits_clean = logits[ind_clean]
    labels_clean = labels[ind_clean]

    if ind_forget.shape[0] > 1:
        # for samples with high loss
        #   high loss, high std --> mislabeling 
        #   high loss, low std  --> irrelevant category
        indices_forget = indices[ind_forget]
        logits_forget = logits[ind_forget]
        pred_distribution = F.softmax(logits_forget, dim=1)
        batch_std = pred_distribution.std(dim=1)

        flag = F.softmax(logits_clean, dim=1).std(dim=1).mean().item()
        # print('{:.5f}'.format(flag), end='*****')
        
        batch_std_sorted, ind_std_sorted = torch.sort(batch_std.data, descending=True)
        ind_split = split_set(batch_std_sorted, flag)
        if ind_split is None:
            ind_split = -1 
        # print('{} == {}'.format(batch_std_sorted, ind_split), end=' ---> ')

        # uncertain could be either mislabeled or hard example
        ind_uncertain = ind_std_sorted[:(ind_split+1)]
        ind_openset = ind_std_sorted[(ind_split+1):]

    else:
        ind_uncertain = 0
        ind_openset = ind_std_sorted[:]


    return ind_clean,ind_uncertain,ind_openset


def split_set(x, flag):
    # split set based in interval
    # x shape is (N), x is sorted in descending
    assert (x > 0).all()
    if x.shape[0] == 1:
        return None
    tmp = (x < flag).nonzero()
    if tmp.shape[0] == 0:
        return None
    else:
        return tmp[0, 0] - 1