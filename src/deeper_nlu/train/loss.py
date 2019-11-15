import torch
import torch.nn.functional as F
import torch.nn as nn
from ..util import lin_comb

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=.1, reduction='mean'):
        super().__init__()
        self.eps, self.reduction = eps, reduction
    
    def forward(self, outp, target):
        c = outp.size()[-1]
        log_preds = F.log_softmax(outp, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return lin_comb(loss/c, nll, self.eps)

def cross_entropy_flat(outp, target, ignore_index=1):
    bs, sl = target.size()
    return F.cross_entropy(outp.view(bs*sl, -1), target.view(bs*sl), ignore_index=ignore_index)

def mlm_loss(outp, target, ignore_index=1):
    pred, x = outp
    target = target[0]
    bs, sl = target.size()
    x_flat, target_flat = x.view(-1), target.clone().view(-1)
    target_flat[x_flat == target_flat] = ignore_index
    return cross_entropy_flat(pred, target_flat.view(bs, -1), ignore_index=ignore_index)

class CombinedLoss(nn.Module):
    def __init__(self, padding_idx=1):
        super().__init__()
        self.padding_idx = padding_idx
    
    def forward(self, outp, target):
        class_outp, seq_outp = outp
        class_target, seq_target = target
        if seq_target.size(1) > seq_outp.size(1): seq_target = seq_target[:,:seq_outp.size(1)].contiguous() # For BERT
        seq_loss = cross_entropy_flat(seq_outp, seq_target, ignore_index=self.padding_idx)
        class_loss = F.cross_entropy(class_outp, class_target)
        return class_loss + seq_loss