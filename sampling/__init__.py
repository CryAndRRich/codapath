from typing import List

__sampler__ = {}


def register_sampler(name: str):
    def wrapper(fn):
        if name in __sampler__:
            raise ValueError(f"Sampler '{name}' is already registered")
        __sampler__[name] = fn
        return fn
    return wrapper


def get_sampler(name: str, **kwargs) -> List[int]:
    if name not in __sampler__:
        raise ValueError(
            f"Sampler '{name}' is not registered. "
            f"Available: {list(__sampler__.keys())}"
        )
    return __sampler__[name](**kwargs)


from . import basic_samplers     
from . import coreset            
from . import typiclust          
from . import activeft           
from . import badge                       
from . import uncertainty_herding 
from . import tcm                
from . import dropquery          
from . import refine     

from . import codapath
from . import scalpel
from . import scalpel_multiscale
from . import nucleus_al
from . import nucleus_coverage
from . import graph_deuce