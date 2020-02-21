import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import tensor, LongTensor
import pandas as pd
from sklearn.model_selection import train_test_split
from ..util import listify, compose
import random
from collections import defaultdict
from functools import partial
from .processor import *
import ujson as json
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __getitem__(self, i):
        return self.df.iloc[i].to_dict('records') if isinstance(i, slice) else self.df.iloc[i].to_dict()
    
    def __len__(self):
        return len(self.df)
    
    def new(self, df, cls=None):
        if cls is None: cls = self.__class__
        return cls(df)
    
    @classmethod
    def from_df(cls, df, tfms=None):
        if tfms is None: tfms = {}
        for k,tfm in tfms.items(): df[k] = df.apply(tfm, axis=1)
        return cls(df)
    
    @classmethod
    def from_csv(cls, path, names=None, *args, **kwargs):
        if names is None: names = ['domain', 'intent', 'annotation', 'profile']
        df = pd.read_csv(path, sep='\t', names=names).fillna('')
        return cls.from_df(df, *args, **kwargs)
    
    @classmethod
    def from_serializable(cls, contents):
        df = pd.read_json(contents)
        return cls.from_df(df)
    
    def to_serializable(self):
        return self.df.to_json()
    
    def split(self, stratify_by=None, train_proportion=.8, *args, **kwargs):
        assert .0 <= train_proportion <= 1., 'Train proportion needs to be between 0 and 1'
        stratify_by = listify(stratify_by)

        train_dfs, val_dfs = [], []
        def _split(_df):
            if train_proportion == 1.: train_df, val_df = _df, pd.DataFrame(columns=_df.columns)
            else: train_df, val_df = train_test_split(_df, train_size=train_proportion, *args, **kwargs)
            train_dfs.append(train_df)
            val_dfs.append(val_df)

        if stratify_by:
            grouped = self.df.groupby(stratify_by)
            for _,g in grouped: _split(g)
        else: _split(self.df)
        
        train_dfs, val_dfs = map(pd.concat, [train_dfs, val_dfs])
        train_ds, val_ds = map(self.new, [train_dfs, val_dfs])
        return SplitDataset(train_ds, val_ds)
    
    def label(self, proc_x=None, proc_y=None):
        if proc_x is None: proc_x = {}
        assert isinstance(proc_x, dict), '`proc_x` needs to be dictionary mapping a field name to one or more processors'
        if proc_y is None: proc_y = {}
        assert isinstance(proc_y, dict), '`proc_y` needs to be dictionary mapping a field name to one or more processors'
        x = [compose(self.df[k], listify(proc)) for k,proc in proc_x.items()]
        y = [compose(self.df[k], listify(proc)) for k,proc in proc_y.items()]
        return LabeledData(x, y)

class LabeledData:
    def __init__(self, x, y):
        self.x, self.y = map(listify, [x, y])
    
    def __getitem__(self, i):
        return list(map(lambda x: x[i], self.x)), list(map(lambda y: y[i], self.y))
    
    def __len__(self):
        return len(self.x[0])
    
    @classmethod
    def from_serializable(cls, contents):
        jsn = json.loads(contents)
        return cls(**jsn)
    
    def to_serializable(self):
        return json.dumps({'x':self.x, 'y':self.y})

class DataBunch:
    def __init__(self, train_dl, valid_dl):
        self.train_dl, self.valid_dl = train_dl, valid_dl
    @property
    def train_ds(self): return self.train_dl.dataset
    @property
    def valid_ds(self): return self.valid_dl.dataset

    @classmethod
    def from_serializable(cls, contents, pad_idx=1):
        sd = SplitDataset.from_serializable(contents)
        return cls(*get_dls(sd.train_ds, sd.valid_ds, pad_idx=pad_idx))
    
    def to_serializable(self):
        return SplitDataset(self.train_dl.dataset, self.valid_dl.dataset).to_serializable()

class EmpiricalSampler(Sampler):
    def __init__(self, data_source, key): self.data_source, self.key = data_source, key
    def __len__(self): return len(self.data_source)
    def __iter__(self):
        counts = tensor(list(map(self.key, self.data_source)))
        counts = counts.sqrt()
        total = counts.sum()
        pvals = counts/total
        draws = np.random.multinomial(total, pvals)
        idxs = torch.cat([tensor([self.data_source[idx]]*count) for idx,count in zip(list(range(len(draws))), draws)])
        idxs = torch.randperm(idxs)
        return iter(idxs)

class SortSampler(Sampler):
    def __init__(self, data_source, key): self.data_source, self.key = data_source, key
    def __len__(self): return len(self.data_source)
    def __iter__(self):
        return iter(sorted(list(range(len(self.data_source))), key=self.key, reverse=True))

class SortishSampler(Sampler):
    def __init__(self, data_source, key, bs):
        self.data_source, self.key, self.bs = data_source, key, bs
    
    def __len__(self):
        return len(self.data_source)
    
    def __iter__(self):
        idxs = torch.randperm(len(self.data_source))
        megabatches = [idxs[i:i+self.bs*50] for i in range(0, len(idxs), self.bs*50)]
        sorted_idx = torch.cat([tensor(sorted(s, key=self.key, reverse=True)) for s in megabatches])
        batches = [sorted_idx[i:i+self.bs] for i in range(0, len(sorted_idx), self.bs)]
        max_idx = torch.argmax(tensor([self.key(ck[0]) for ck in batches])) # find chunk with largest key
        batches[0], batches[max_idx] = batches[max_idx], batches[0] # make sure it goes first
        batch_idxs = torch.randperm(len(batches)-2)
        sorted_idx = torch.cat([batches[i+1] for i in batch_idxs]) if len(batches) > 1 else LongTensor([])
        sorted_idx = torch.cat([batches[0], sorted_idx, batches[-1]])
        return iter(sorted_idx)

def pad_collate(samples, pad_idx=1, pad_first=False):
    max_len = max([len(x[0]) for x,y in samples])
    bs = len(samples)
    x,y = samples[0]
    inps = [torch.zeros(bs, max_len).long() + pad_idx if isinstance(_x, list) else torch.zeros(bs).long() for _x in x]
    outps = [torch.zeros(bs, max_len).long() + pad_idx if isinstance(_y, list) else torch.zeros(bs).long() for _y in y]
    for i,(x,y) in enumerate(samples):
        for j,_x in enumerate(x):
            if isinstance(_x, list):
                if pad_first: inps[j][i, -len(_x):] = LongTensor(_x)
                else: inps[j][i, :len(_x)] = LongTensor(_x)
            else: inps[j][i] = _x
        for j,_y in enumerate(y):
            if isinstance(_y, list):
                if pad_first: outps[j][i, -len(_y):] = LongTensor(_y)
                else: outps[j][i, :len(_y)] = LongTensor(_y)
            else: outps[j][i] = _y
    return inps, outps

def get_dls(train_ds, valid_ds, bs=64, pad_idx=1, train_sampler=None, valid_sampler=None, collate_func=None, *args, **kwargs):
    train_sampler = SortishSampler(train_ds.x[0], key=lambda t: len(train_ds.x[0][t]), bs=bs) if train_sampler is None else train_sampler(train_ds)
    valid_sampler = SortSampler(valid_ds.x[0], key=lambda t: len(valid_ds.x[0][t])) if valid_sampler is None else valid_sampler(valid_ds)
    collate_fn = collate_func or partial(pad_collate, pad_idx=pad_idx)
    return (
        DataLoader(train_ds, batch_size=bs, sampler=train_sampler, collate_fn=collate_fn, **kwargs),
        DataLoader(valid_ds, batch_size=bs*2, sampler=valid_sampler, collate_fn=collate_fn, **kwargs)
    )

class SplitDataset:
    def __init__(self, train_ds, valid_ds):
        self.train_ds, self.valid_ds = train_ds, valid_ds
        self.dataset_class_name = train_ds.__class__.__name__
    
    def new(self, train_ds, valid_ds, cls=None):
        if cls is None: cls = self.__class__
        return cls(train_ds, valid_ds)
    
    def label(self, *args, **kwargs):
        return self.new(self.train_ds.label(*args, **kwargs), self.valid_ds.label(*args, **kwargs))
    
    def to_databunch(self, *args, **kwargs):
        assert isinstance(self.train_ds, LabeledData), 'First label your data'
        return DataBunch(*get_dls(self.train_ds, self.valid_ds, *args, **kwargs))
    
    @classmethod
    def from_serializable(cls, contents):
        jsn = json.loads(contents)
        from_serializable = eval('{}.from_serializable'.format(jsn['dataset_class_name']))
        train_ds, valid_ds = map(from_serializable, [jsn['train_ds'], jsn['valid_ds']])
        return cls(train_ds, valid_ds)
    
    def to_serializable(self):
        return json.dumps({
            'train_ds': self.train_ds.to_serializable(), 
            'valid_ds': self.valid_ds.to_serializable(), 
            'dataset_class_name': self.dataset_class_name
        })