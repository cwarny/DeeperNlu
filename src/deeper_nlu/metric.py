import torch

def accuracy(out, yb): return (torch.argmax(out, dim=-1)==yb).float().mean()

def accuracy_flat(outp, target):
    bs,sl = target.size()
    return accuracy(outp.view(bs * sl, -1), target.view(bs * sl))

def combined_accuracy(outp, target, padding_idx=1):
    class_outp, seq_outp = outp
    class_target, seq_target = target
    if seq_target.size(1) > seq_outp.size(1): seq_target = seq_target[:,:seq_outp.size(1)].contiguous() # For BERT
    bs = class_target.size(0)
    best_class = torch.argmax(class_outp, dim=-1)
    correct_class = torch.eq(best_class, class_target)
    best_seq = torch.argmax(seq_outp, dim=-1)
    correct_seq_items = torch.eq(best_seq, seq_target)
    valid_seq_items = torch.ne(seq_target, padding_idx)
    correct_seq = (correct_seq_items * valid_seq_items).sum(1) == valid_seq_items.sum(1)
    n_correct = (correct_seq & correct_class).sum().float()
    return (n_correct / bs * 100).item()

def seq_acc(outp, target, padding_idx=1):
    if target.size(1) > outp.size(1): target = target[:,:outp.size(1)].contiguous() # For BERT
    bs = target.size(0)
    best = torch.argmax(outp, dim=-1)
    correct_items = torch.eq(best, target)
    valid_items = torch.ne(target, padding_idx)
    correct = (correct_items * valid_items).sum(1) == valid_items.sum(1)
    n_correct = correct.sum().float()
    return (n_correct / bs * 100).item()

def seq_accuracy(outp, target, **kwargs):
    class_outp, seq_outp = outp
    class_target, seq_target = target
    return seq_acc(seq_outp, seq_target, **kwargs)

def class_accuracy(outp, target):
    class_outp, seq_outp = outp
    class_target, seq_target = target
    acc = accuracy(class_outp, class_target)
    return (acc*100).item()

def mlm_accuracy(outp, target):
    pred, x = outp
    target = target[0]
    bs, sl = target.size()
    x_flat, target_flat, pred_flat = x.view(-1), target.view(-1), pred.view(bs*sl,-1)
    selector = x_flat == target_flat
    return accuracy(pred_flat[selector], target_flat[selector])