"""Published baselines, reproduced against their papers and reference code.

See `main.py` for how each one is swept over a budget list, and the module
docstring of each file for the paper/repo it was verified against.

`dropquery` is imported but is NOT a runnable baseline: it has no entry in
`specs.SAMPLER_SPECS`, so `spec_for` refuses it and it cannot be swept. It
stays because `refine`'s candidate-generation ensemble calls it by name
through the registry -- dropping the module would silently change `refine`
away from its published five-strategy configuration.
"""

from . import (
    activeft,
    badge,
    basic,
    coreset,
    dropquery,
    refine,
    typiclust,
    uncertainty_herding,
)

__all__ = [
    "activeft",
    "badge",
    "basic",
    "coreset",
    "dropquery",
    "refine",
    "typiclust",
    "uncertainty_herding",
]
