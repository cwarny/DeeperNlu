from functools import partial
import math
import torch
from torch import tensor
from ..util import listify

def annealer(f):
    def _inner(start, end, pos):
        return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos):
    return start + pos*(end-start)

@annealer
def sched_cos(start, end, pos):
    return start + (1+math.cos(math.pi*(1-pos))) * (end-start) / 2

@annealer
def sched_no(start, end, pos):
    return start

@annealer
def sched_exp(start, end, pos):
    return start * (end/start) ** pos

@annealer
def cos_1cycle_anneal(start, end, pos):
    pass

def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1
    pcts = tensor([0] + pcts)
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max() # `.nonzero()` return the indices of the nonzero items or in this case the indices of the percentages that are lower or equal to the current value (`pos`). `.max()` will give the index of the highest percentage. So basically this return the index of the closest percentage
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx]) # prorated
        return scheds[idx](actual_pos)
    return _inner

def create_phases(phases):
    phases = listify(phases)
    return phases*[1-sum(phases)]