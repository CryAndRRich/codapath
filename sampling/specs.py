"""How each sampler must be swept over a list of budgets.

Two INDEPENDENT properties decide this, and conflating them has caused real
bugs in this project more than once. They are kept as separate fields here so
that a reader never has to infer one from the other:

`passes` — a property of the ALGORITHM, from its paper.
    "single" : one selection pass produces the whole batch.
    "multi"  : the method interleaves selection with retraining a probe, so it
               runs several internal rounds for one budget.

`prefix_exact` — a property of the IMPLEMENTATION's dependence on the target
budget B. True means: running once at max(budgets) and taking the first B
picks yields exactly what running directly at B would have produced, so the
sweep can share one run. False means every budget needs its own run.

The two are genuinely orthogonal, and must stay separate fields even though
no CURRENT sampler is multi-round-and-prefix-exact: `tcm` was that case and
has been removed, so the combination is unpopulated rather than impossible.
`typiclust` and `activeft` are single-pass yet NOT prefix-exact, because their
one pass reads B directly (`typiclust` sets `num_clusters` from it, `activeft`
parameterises the optimisation by it), so a large-B run is not a superset of a
small-B run.

Anything that scales an internal threshold by B is not prefix-exact, however
"one-shot" it looks. That is why `uncertainty_herding` is False (its coverage/
uncertainty phase switch sits at 0.2*B) and why `refine` is False too (its
stage-2 head IS Uncertainty Herding, so it inherits the same problem).

`needs` lists the extra pool arrays a sampler requires beyond the common
`image_embeddings`/`oracle_labels`/`num_classes`/`max_budget`/`device`. It is
NOT a classification axis — an earlier version of this file split the samplers
into three sets where two of them differed only by this field, which read as a
third category that does not exist.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class SamplerSpec:
    passes: str
    prefix_exact: bool
    needs: Tuple[str, ...] = ()
    prefix_exact_min_class_multiple: int = 0
    why: str = ""

    def is_prefix_exact(self, budget: int, num_classes: int) -> bool:
        """Prefix-exact for THIS budget, including any lower bound."""
        if not self.prefix_exact:
            return False
        return budget >= self.prefix_exact_min_class_multiple * num_classes


SAMPLER_SPECS: Dict[str, SamplerSpec] = {
    # --- single pass, prefix-exact: one greedy order serves every budget ---
    "random": SamplerSpec("single", True, why="uniform draw; a prefix is still uniform"),
    "coreset": SamplerSpec("single", True, why="k-center greedy order is budget-free"),
    # --- single pass, NOT prefix-exact: the single pass reads B ---
    "typiclust": SamplerSpec(
        "single", False, why="num_clusters = min(labeled + B, MAX) depends on B",
    ),
    "activeft": SamplerSpec(
        "single", False, why="optimises a parameterisation defined by B",
    ),
    # --- multi round, NOT prefix-exact: each budget needs its own run ---
    "margin": SamplerSpec("multi", False, why="probe is retrained per round"),
    "entropy": SamplerSpec("multi", False, why="probe is retrained per round"),
    "badge": SamplerSpec("multi", False, why="gradient embeddings follow the probe"),
    "uncertainty_herding": SamplerSpec(
        "multi", False, why="coverage/uncertainty phase switch sits at 0.2*B",
    ),
    "refine": SamplerSpec(
        "multi", False, why="stage-2 head is Uncertainty Herding, same 0.2*B switch",
    ),
    "pact": SamplerSpec(
        "multi", False, needs=("cell_embeddings", "cell_reliability"),
        why="probes are retrained and sigma re-adapts every round",
    ),
}

# Every sampler that needs neither a CellViT cell view nor a VLM text prior, so
# it can run from the DINOv2 visual cache alone. Derived rather than hand-listed
# a second time, so it cannot drift out of sync with SAMPLER_SPECS itself.
BASELINE_SAMPLERS = frozenset(
    name for name in SAMPLER_SPECS if name != "pact"
)


def spec_for(name: str) -> SamplerSpec:
    if name not in SAMPLER_SPECS:
        raise ValueError(
            f"No sweep spec for sampler '{name}'. Add one to SAMPLER_SPECS: a new "
            f"sampler must state its passes/prefix_exact explicitly rather than "
            f"fall back to a default that may silently be wrong."
        )
    return SAMPLER_SPECS[name]
