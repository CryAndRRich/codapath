"""Published baselines, reproduced against their papers and reference code.

See `main.py` for how each one is swept over a budget list, and the module
docstring of each file for the paper/repo it was verified against.
"""

from . import (
    activeft,
    badge,
    basic,
    codapath,
    coreset,
    dropquery,
    refine,
    tcm,
    typiclust,
    uncertainty_herding,
)

__all__ = [
    "activeft",
    "badge",
    "basic",
    "codapath",
    "coreset",
    "dropquery",
    "refine",
    "tcm",
    "typiclust",
    "uncertainty_herding",
]
