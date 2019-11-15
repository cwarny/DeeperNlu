from typing import Iterable
from concurrent.futures import ProcessPoolExecutor

def get_xavier_vector(size):
    vec = torch.ones(1, size)
    torch.nn.init.xavier_uniform_(vec)
    return vec

def find_modules(m, cond):
    if cond(m):
        return [m]
    return sum([find_modules(o, cond) for o in m.children()], [])

def compose(x, funcs, *args, order_key='_order', **kwargs):
    # `compose` passes to each function it composes the same keyword args passed to it
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key): x = f(x, **kwargs)
    return x

def listify(x, max_depth=0, d=0):
    if x is None: return []
    if isinstance(x, list):
        if d < max_depth: return [listify(e, d=d+1) for e in x]
        return x
    if isinstance(x, str): return [x]
    if isinstance(x, Iterable): return list(x)
    return [x]

def get_batch(dl, run):
    run.xb, run.yb = next(iter(dl))
    for cb in run.cbs:
        cb.set_runner(run)
    run('begin_batch')
    return run.xb, run.yb

def lin_comb(a, b, alpha):
    return a*alpha + b*(1-alpha)

def unsqueeze(inp, dims):
    for dim in listify(dims): inp = torch.unsqueeze(inp, dim)
    return inp

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

def read_file(fn, split_lines=False):
    with open(fn, 'r', encoding='utf8') as f: 
        return f.readlines() if split_lines else f.read()

def parallel(func, arr, max_workers=4):
    if max_workers<2:
        results = list(map(func, enumerate(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(func, enumerate(arr)))
    if any([o is not None for o in results]):
        return results