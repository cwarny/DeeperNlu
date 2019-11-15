import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt
import time
from pathlib import Path
from ..util import unsqueeze, reduce_loss, listify
from ..exception import CancelTrainException

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

class Callback:
    _order = 0
    def set_runner(self, run): self.run = run
    
    def __getattr__(self, k): return getattr(self.run, k) # Delegate to run any attempt at accessing an attribute not defined in this class
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
    
    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True # if the callback method exists and returns something truthy, return True
        return False

class TrainEvalCallback(Callback):
    _order = 0
    def begin_fit(self):
        self.run.n_epochs = 0.
        self.run.n_iter = 0
    
    def after_batch(self):
        if not self.in_train: return # don't do anything
        self.run.n_epochs += 1./self.iters
        self.run.n_iter += 1
    
    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True
    
    def begin_validate(self):
        self.model.eval()
        self.run.dl = self.data.valid_dl
        self.run.in_train = False

class EarlyStop(Callback):
    def __init__(self, patience=4):
        super().__init__()
        self.valid_stats = AvgStats(None, False)
        self.avg_loss_prev, self.avg_loss_curr = None, None
        self.patience, self.frustration = patience, 0
    
    def begin_epoch(self): self.valid_stats.reset()
    
    def after_loss(self):
        if not self.in_train:
            with torch.no_grad(): self.valid_stats.accumulate(self.run)
    
    def after_epoch(self):
        self.avg_loss_prev = self.avg_loss_curr
        self.avg_loss_curr = self.valid_stats.avg_stats[0]
        if self.avg_loss_prev is not None:
            if self.avg_loss_curr <= self.avg_loss_prev: self.frustration = 0
            else: self.frustration += 1
        if self.frustration >= self.patience:
            raise CancelTrainException()

class AvgStats:
    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train
    
    def reset(self):
        self.tot_loss, self.count = 0, 0
        self.tot_mets = [0.] * len(self.metrics)
    
    @property
    def all_stats(self):
        return [self.tot_loss if isinstance(self.tot_loss, int) else self.tot_loss.item()] + self.tot_mets
    
    @property
    def avg_stats(self):
        if self.count: return [o/self.count for o in self.all_stats]
        else: return [float('nan') for o in self.all_stats]
    
    def __repr__(self):
        if not self.count:
            return ''
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"
    
    def accumulate(self, run):
        batch_size = run.xb[0].shape[0] if isinstance(run.xb, list) else run.xb.shape[0]
        self.tot_loss += run.loss * batch_size
        self.count += batch_size
        for i,metric in enumerate(self.metrics):
            self.tot_mets[i] += metric(run.pred, run.yb) * batch_size

def format_duration(d):
    return str(d)

def get_function_name(func):
    try: return func.__name__
    except AttributeError: return func.func.__name__

class AvgStatsCallback(Callback):
    _order = 1
    def __init__(self, metrics, path=None):
        super().__init__()
        self.train_stats, self.valid_stats = AvgStats(metrics, True), AvgStats(metrics, False)
        if path is not None: self.path = Path(path)
        else: self.path = path
    
    def begin_fit(self):
        met_names = ['loss'] + [get_function_name(m) for m in self.train_stats.metrics]
        names = ['epoch'] + [f"train_{n}" for n in met_names] + [f"valid_{n}" for n in met_names] + ['time']
        self.logger(names)
        if self.path is not None: self.path.write_text('\t'.join(names) + '\n')

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()
    
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    
    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f"{v:.6f}" for v in o.avg_stats]
        stats += [format_duration(time.time() - self.start_time)]
        self.logger(stats)
        if self.path is not None: self.path.open('a').write('\t'.join(map(str, stats)) + '\n')

class LR_Find(Callback):
    _order = 3

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        super().__init__()
        self.max_iter, self.min_lr, self.max_lr = max_iter, min_lr, max_lr
        self.best_loss = 1e9
    
    def begin_batch(self):
        # self.in_train will be delegated to the runner
        if not self.in_train: return # don't do anything if you're in the training phase
        pos = self.n_iter/self.max_iter # position
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos # exponential curve to set the learning rate
        for pg in self.opt.hypers: pg['lr'] = lr
    
    def after_step(self):
        if self.n_iter >= self.max_iter or self.loss > self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss:
            self.best_loss = self.loss

class Recorder(Callback):
    _order = 4
    def begin_fit(self):
        self.lrs, self.losses = [], []
    
    def after_batch(self):
        if not self.in_train: return
        self.lrs.append(self.opt.hypers[-1]['lr'])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr(self): plt.plot(self.lrs)
    
    def plot_loss(self): plt.plot(self.losses)

    def plot(self, skip_last=0):
        losses = [o.item() for o in self.losses]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(self.lrs[:n], losses[:n])

class KaimingInitializer(Callback):
    _order = -1
    def begin_fit(self):
        for layer in self.run.model:
            nn.init.kaiming_uniform_(layer[0].weight)
            layer[0].bias.data.zero_()

class ParamScheduler(Callback):
    _order = 5
    def __init__(self, pname, sched_func):
        super().__init__()
        self.pname, self.sched_func = pname, sched_func
    
    def set_param(self):
        for pg in self.opt.hypers:
            pg[self.pname] = self.sched_func(self.n_epochs/self.epochs)
        
    def begin_batch(self):
        if self.in_train: self.set_param()

class CudaCallback(Callback):
    _order = 1
    def __init__(self, device):
        self.device = device
        super().__init__()
    
    def begin_fit(self):
        self.model.to(self.device)
    
    def begin_batch(self):
        self.run.xb = list(map(lambda x: x.to(self.device), self.xb)) if isinstance(self.xb, list) else self.xb.to(self.device)
        self.run.yb = list(map(lambda y: y.to(self.device), self.yb)) if isinstance(self.yb, list) else self.yb.to(self.device)

class BatchTransformCallback(Callback):
    '''Applies transformation to input batch'''
    _order = 2
    
    def __init__(self, tfm):
        self.tfm = tfm
        super().__init__()
    
    def begin_batch(self):
        self.run.xb = self.tfm(self.xb)

class NoneReduce:
    def __init__(self, loss_func):
        self.loss_func, self.old_red = loss_func, None
    
    def __enter__(self):
        if hasattr(self.loss_func, 'reduction'):
            self.old_red = getattr(self.loss_func, 'reduction')
            setattr(self.loss_func, 'reduction', 'none')
            return self.loss_func
        else:
            return partial(self.loss_func, reduction='none')
    
    def __exit__(self, typ, value, traceback):
        if self.old_red is not None:
            setattr(self.loss_func, 'reduction', self.old_red)

class GradientClipping(Callback):
    def __init__(self, clip=None): 
        super().__init__()
        self.clip = clip
    def after_backward(self):
        if self.clip: nn.utils.clip_grad_norm_(self.run.model.parameters(), self.clip)

class SaveModel(Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.valid_stats = AvgStats(None, False)
        self.avg_loss_prev, self.avg_loss_curr, self.avg_loss_best = None, None, float('inf')
    
    def begin_epoch(self):
        self.valid_stats.reset()
    
    def after_loss(self):
        if not self.in_train:
            with torch.no_grad(): self.valid_stats.accumulate(self.run)
    
    def after_epoch(self):
        self.avg_loss_prev = self.avg_loss_curr
        self.avg_loss_curr = self.valid_stats.avg_stats[0]
        if self.avg_loss_prev is not None and self.avg_loss_curr <= self.avg_loss_prev: # if loss got better
            if self.avg_loss_curr < self.avg_loss_best: # if our loss is less than the best loss recorded so far, overwrite and save
                torch.save(self.model.state_dict(), self.path)
                self.avg_loss_best = self.avg_loss_curr

class Mixup(Callback):
    # This callback will change the loss function
    _order = 90
    def __init__(self, alpha=.2):
        self.distrib = Beta(tensor([alpha]), tensor([alpha]))
    
    def begin_fit(self):
        self.old_loss_func, self.run.loss_func = self.run.loss_func, self.loss_func

    def begin_batch(self):
        if not self.in_train: return # Only mixup during training
        bs = self.yb.size(0)
        dev = self.xb.device
        lamb = self.distrib.sample((bs,)).squeeze().to(dev)
        lamb = torch.stack([lamb, 1-lamb], 1)
        self.lamb = unsqueeze(lamb.max(1)[0], (1,2,3))
        shuffle = torch.randperm(bs).to(dev) 
        xb1, self.yb1 = self.xb[shuffle], self.yb[shuffle]
        self.run.xb = lin_comb(self.xb, xb1, self.lamb)
    
    def after_fit(self):
        self.run.loss_func = self.old_loss_func
    
    def loss_func(self, pred, yb):
        if not self.in_train: return self.old_loss_func(pred, yb)
        with NoneReduce(self.old_loss_func) as loss_func:
            # We need to compute the loss for a new input which
            # is the lin comb of inputs A (original inputs) with 
            # inputs B (some other inputs to linearly combine A with,
            # using linear combination parameter lambda). 
            # The natural thing to do would be to use as target a 
            # dense target where instead of a one-hot vector with a 1
            # for the right label and a 0 everywhere else, we would
            # use lambda for the labels corresponding to the original
            # inputs A and 1-lambda for the labels corresponding
            # to the inputs B. However, that would be cumbersome and
            # computationally intensive.
            # Thankfully, we can prove that computing the loss func 
            # for the lin comb of the inputs A with inputs B by using 
            # dense target is equivalent to computing it by using the 
            # same lin comb of, on the one hand, the loss as computed
            # with the old labels, and on the other hand, the loss
            # as computed with the labels corresponding to the
            # inputs the original inputs were linearly combined with.
            loss1 = loss_func(pred, yb) # Calculate the loss on the regular set of images
            loss2 = loss_func(pred, self.yb1) # Calculate the loss on the reshuffled set of labels
        loss = lin_comb(loss1, loss2, self.lamb)
        return reduce_loss(loss, getattr(self.old_loss_func, 'reduction', 'mean'))

# TODO: 
# 1. Mixup: linearly interpolate two inputs on the embeddings and also linearly interpolate the associated labels
# 2. Affine transformations on embedding layer.