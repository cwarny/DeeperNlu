from functools import partial
from ..util import listify

class Hook:
    def __init__(self, module, func):
        self.hook = module.register_forward_hook(partial(func, self)) # the callback passed to register_forward_hook will be called every time the module is calculated
    
    def remove(self):
        self.hook.remove()
    
    def __del__(self):
        self.remove()

class Hooks:
    def __init__(self, modules, func):
        super().__init__([Hook(module, func) for module in listify(modules)])
    
    def __enter__(self, *args):
        return self
    
    def __exit__(self, *args):
        self.remove()
    
    def __del__(self): # automatically called by python when doing memory cleanup
        self.remove()
    
    def __delitem__(self, idx):
        self[idx].remove()
        super().__delitem__(idx)
    
    def remove(self):
        for hook in self:
            hook.remove()