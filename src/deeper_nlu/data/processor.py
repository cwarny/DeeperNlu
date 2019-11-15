from ..util import parallel, listify
from collections import defaultdict, Counter
import json
import random
import itertools

def default_tokenizer(t): return t.split()

class Processor:
    def __call__(self, items):
        return self.process(items)

    def proc1(self, item):
        return item
    
    def process(self, items): 
        return [self.proc1(item) for item in items]
    
    def deproc1(self, item):
        return item

    def deprocess(self, items):
        return [self.deproc1(item) for item in items]

class TokenizeProcessor(Processor):
    def __init__(self, tokenizer=None, chunksize=2000, max_workers=4):
        self.chunksize, self.max_workers = chunksize, max_workers
        self.tokenizer = default_tokenizer if tokenizer is None else tokenizer

    def __call__(self, items):
        chunks = [items[i:i+self.chunksize] for i in (range(0, len(items), self.chunksize))]
        toks = parallel(self.proc_chunk, chunks, self.max_workers)
        return list(itertools.chain.from_iterable(toks))
    
    def proc_chunk(self, args):
        _,chunk = args
        return self.process(chunk)
    
    def proc1(self, text):
        return self.tokenizer(text)
    
    def deproc1(self, tokens):
        return ' '.join(tokens)

class NumericalizeProcessor(Processor):
    def __init__(self, vocab=None, max_vocab=60000, min_freq=2, unk_idx=0, special_tokens=None):
        self.vocab, self.unk_idx, self.max_vocab, self.min_freq = vocab, unk_idx, max_vocab, min_freq
        if special_tokens is not None: assert isinstance(special_tokens, dict), 'special_tokens needs to be a dict mapping the special token to an index'
        else: special_tokens = {}
        self.special_tokens = list(special_tokens.items())

    def __call__(self, items):
        if self.vocab is None:
            freq = Counter(p for o in items for p in o)
            self.vocab = [o for o,c in freq.most_common(self.max_vocab) if c >= self.min_freq]
            for tok,idx in sorted(self.special_tokens, key=lambda d:-d[1]):
                if tok in self.vocab: self.vocab.remove(tok)
                self.vocab.insert(idx, tok)
        if getattr(self, 'otoi', None) is None:
            self.otoi = defaultdict(lambda: self.unk_idx, {v:k for k,v in enumerate(self.vocab)}) # Returns self.unk_idx if can't find
        return self.process(items)

    def proc1(self, tokens):
        return [self.otoi[tok] for tok in tokens]
    
    def deproc1(self, idx):
        return [self.vocab[i] for i in idx]

    def save(self, path):
        with open(path, 'w') as f: json.dump({'vocab':self.vocab, 'unk_idx':self.unk_idx}, f)
    
    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            contents = json.load(f)
            return cls(**contents)

class CategoryProcessor(Processor):
    def __init__(self, vocab=None, min_freq=2, special_categories='Other', unk_idx=0):
        self.vocab, self.unk_idx, self.min_freq, self.special_categories = vocab, unk_idx, min_freq, listify(special_categories)
    
    def __call__(self, items):
        if self.vocab is None:
            freq = Counter(o for o in items)
            self.vocab = [o for o,c in freq.most_common() if c >= self.min_freq]
            for cat in reversed(self.special_categories):
                if cat in self.vocab: self.vocab.remove(cat)
                self.vocab.insert(self.unk_idx, cat) # add special categories starting at index `self.unk_idx`
        if getattr(self, 'otoi', None) is None:
            self.otoi  = defaultdict(lambda: self.unk_idx, {v:k for k,v in enumerate(self.vocab)}) # Returns 0 if can't find
        return self.process(items)
    
    def proc1(self, item): 
        return self.otoi[item]
    
    def deproc1(self, idx):
        return self.vocab[idx]
    
    def save(self, path):
        with open(path, 'w') as f: json.dump({'vocab':self.vocab, 'unk_idx':self.unk_idx}, f)
    
    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            contents = json.load(f)
            return cls(**contents)

class MaskProcessor(Processor):
    def __init__(self, mask_tok, mask_rate=.15, bos=True, eos=True):
        self.mask_tok, self.mask_rate, self.bos, self.eos = mask_tok, mask_rate, bos, eos
    
    def proc1(self, tokens):
        length = len(tokens)
        if self.eos: length -= 1
        if self.bos: length -= 1
        n = max(int(length*self.mask_rate + 0.5), 1)
        idxs = list(range(length))
        sample = random.sample(idxs, n)
        for idx in sample: 
            if self.bos: idx += 1
            tokens[idx] = self.mask_tok
        return tokens