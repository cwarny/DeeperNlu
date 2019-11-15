import torch
from torch import tensor
import torch.nn as nn
from .optimizer import sgd_opt
from .callback import TrainEvalCallback
from ..util import listify, find_modules
from ..exception import CancelTrainException, CancelEpochException, CancelBatchException
from .callback import *
from .loss import *
from .optimizer import *
from .schedule import *

def param_getter(m):
    return m.parameters()

class Learner:
    def __init__(self, model, data, loss_func, opt_func=sgd_opt, lr=1e-2, splitter=param_getter, cbs=None, cb_funcs=None):
        self.model, self.data, self.loss_func, self.opt_func, self.lr, self.splitter = model, data, loss_func, opt_func, lr, splitter
        self.in_train, self.logger, self.opt = False, print, None

        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))
    
    def add_cbs(self, cbs):
        for cb in listify(cbs): self.add_cb(cb)
    
    def add_cb(self, cb):
        cb.set_runner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)
    
    def remove_cbs(self, cbs):
        for cb in listify(cbs): self.cbs.remove(cb)
    
    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            self.xb, self.yb = xb, yb
            self('begin_batch')
            self.pred = self.model(self.xb)
            self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:
            self('after_cancel_batch')
        finally:
            self('after_batch')
    
    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i, (xb,yb) in enumerate(self.dl):
                self.one_batch(i, xb, yb)
        except CancelEpochException:
            self('after_cancel_epoch')
    
    def do_begin_fit(self, epochs):
        self.epochs, self.loss = epochs, tensor(0.)
        for cb in self.cbs: cb.set_runner(self)
        self('begin_fit')
    
    def do_begin_epoch(self, epoch):
        self.epoch, self.dl = epoch, self.data.train_dl
        return self('begin_epoch')

    def fit(self, epochs, cbs=None, reset_opt=False):
        self.add_cbs(cbs)
        if reset_opt or not self.opt:
            self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)
        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):
                self.do_begin_epoch(epoch)
                if not self('begin_epoch'): self.all_batches()
                with torch.no_grad():
                    if not self('begin_validate'): self.all_batches()
                self('after_epoch')
        except CancelTrainException:
            self('after_cancel_train')
        finally:
            self('after_fit')
    
    ALL_CBS = {
        'begin_batch',
        'after_pred',
        'after_loss',
        'after_backward',
        'after_step',
        'after_cancel_batch',
        'after_batch',
        'after_cancel_epoch',
        'begin_fit',
        'begin_epoch',
        'begin_validate',
        'after_epoch',
        'after_cancel_train',
        'after_fit'
    }
    
    def __call__(self, cb_name):
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order): # Go through all callbacks in order
            res = cb(cb_name) and res # if this callback object has this callback method 
        return res

from .hook import Hooks, Hook
from ..util import get_batch, find_modules

def model_summary(dl, run, learn, find_all=False, print_mods=True):
    xb, yb = get_batch(dl, run)
    device = next(learn.model.parameters()).device
    xb, yb = xb.to(device), yb.to(device)
    mods = find_modules(learn.model) if find_all else learn.model.children()
    def f(hook, mod, inp, outp): 
        if print_mods: print(f"{mod},\n{outp.shape}\n")
        else: print(f"{outp.shape}\n")
    with Hooks(mods, f) as hooks:
        learn.model(xb)

def lsuv_module(mdl, m, xb):
    '''
    Changes module m bias and weights init until they respectively reach zero mean and unit variance.
    Meant to be run once, before training starts.
    '''
    def append_stat(hook, mod, inp, outp):
        d = outp.data
        hook.mean, hook.std = d.mean().item(), d.std().item()

    h = Hook(m, append_stat)
    # Iterate through a batch until module m has a bias 
    # (defined as the bias of the module's relu) of zero 
    # and a weight (defined as the module's weights) of 1
    while mdl(xb) is not None and abs(h.mean) > 1e-3: # keep looping as long as mean is not close to zero
        m.bias -= h.mean
    while mdl(xb) is not None and abs(h.std-1) > 1e-3:
        m.weight.data /= h.std
    h.remove()
    return h.mean, h.std