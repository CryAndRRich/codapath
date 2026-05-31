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


# ── Basic / classic samplers (random, margin, entropy) ──────────────────────
from . import basic_samplers     

# ── Diversity / coverage samplers ────────────────────────────────────────────
from . import coreset            
from . import typiclust          
from . import activeft           

# ── Gradient-based sampler ────────────────────────────────────────────────────
from . import badge              

# ── Pathology VLM sampler ─────────────────────────────────────────────────────
from . import codapath           

# ── New hybrid / ensemble samplers ───────────────────────────────────────────
from . import uncertainty_herding
from . import dcom               
from . import tcm                
from . import dropquery          
from . import refine             
