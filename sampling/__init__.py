"""Active-learning sample selection.

`baselines` holds published methods reproduced for comparison; `pact` is
this project's own method. Importing this package registers all of them.
"""

from .registry import available_samplers, get_sampler, register_sampler
from . import baselines, pact

__all__ = ["available_samplers", "get_sampler", "register_sampler", "baselines", "pact"]
