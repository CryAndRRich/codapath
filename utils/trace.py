"""Per-step record of what a sampler did, for auditing and for visualisation.

A sampler returns a list of indices. That is enough to train a probe and
nothing else: it cannot say whether the greedy actually discriminated between
candidates, which round each point came from, or how the acquisition score was
distributed over the pool. All of that is computed during selection and then
thrown away.

`SelectionTrace` collects it. Two distinct uses:

* auditing — a run whose per-step scores are all equal, or whose picks arrive
  in increasing index order, has silently degenerated (a constant kernel makes
  every marginal gain zero and `argmax` then returns index order). This is the
  failure mode that "returns N unique indices" tests cannot see.
* visualisation — round, rank, score and the pool score distribution are what
  a later plot needs, and they cannot be recovered after the run.

Every field is optional: a sampler records what it has. Nothing here changes
selection, so a traced run picks exactly the same points as an untraced one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class StepRecord:
    """One selected point."""

    index: int
    round_index: int
    rank: int
    score: Optional[float] = None
    margin_to_runner_up: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoundRecord:
    """One selection round: the state the picks were made under."""

    round_index: int
    num_selected: int
    seconds: float
    sigma: Optional[float] = None
    weight_summary: Optional[Dict[str, float]] = None
    score_summary: Optional[Dict[str, float]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def summarize(values: np.ndarray, quantiles=(0.0, 0.25, 0.5, 0.75, 1.0)) -> Dict[str, float]:
    """Compact distribution summary — a full pool vector is too big to store.

    `spread` is the headline number: a near-zero spread over a non-trivial pool
    means the score carried no information and the pick was effectively
    arbitrary, whatever the mean happens to be.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0.0}
    summary = {
        "count": float(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "spread": float(finite.max() - finite.min()),
        "nonfinite": float(values.size - finite.size),
    }
    for quantile, value in zip(quantiles, np.quantile(finite, quantiles)):
        summary[f"q{int(quantile * 100):02d}"] = float(value)
    return summary


class SelectionTrace:
    """Accumulate per-step and per-round records during one sampler call."""

    def __init__(self, sampler: str, budget: int, pool_size: int) -> None:
        self.sampler = sampler
        self.budget = int(budget)
        self.pool_size = int(pool_size)
        self.steps: List[StepRecord] = []
        self.rounds: List[RoundRecord] = []
        self._round_index = 0

    def start_round(self, round_index: int) -> None:
        self._round_index = int(round_index)

    def add_step(
        self,
        index: int,
        score: Optional[float] = None,
        margin_to_runner_up: Optional[float] = None,
        **extra: float,
    ) -> None:
        self.steps.append(StepRecord(
            index=int(index),
            round_index=self._round_index,
            rank=len(self.steps),
            score=None if score is None else float(score),
            margin_to_runner_up=(
                None if margin_to_runner_up is None else float(margin_to_runner_up)
            ),
            extra={key: float(value) for key, value in extra.items()},
        ))

    def add_round(
        self,
        num_selected: int,
        seconds: float,
        sigma: Optional[float] = None,
        weights: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        **diagnostics: Any,
    ) -> None:
        self.rounds.append(RoundRecord(
            round_index=self._round_index,
            num_selected=int(num_selected),
            seconds=float(seconds),
            sigma=None if sigma is None else float(sigma),
            weight_summary=None if weights is None else summarize(weights),
            score_summary=None if scores is None else summarize(scores),
            diagnostics=dict(diagnostics),
        ))

    def to_payload(self) -> Dict[str, Any]:
        """Plain-dict form, safe for `torch.save` with `weights_only=False`."""
        return {
            "sampler": self.sampler,
            "budget": self.budget,
            "pool_size": self.pool_size,
            "steps": [
                {
                    "index": step.index,
                    "round_index": step.round_index,
                    "rank": step.rank,
                    "score": step.score,
                    "margin_to_runner_up": step.margin_to_runner_up,
                    **step.extra,
                }
                for step in self.steps
            ],
            "rounds": [
                {
                    "round_index": record.round_index,
                    "num_selected": record.num_selected,
                    "seconds": record.seconds,
                    "sigma": record.sigma,
                    "weight_summary": record.weight_summary,
                    "score_summary": record.score_summary,
                    **record.diagnostics,
                }
                for record in self.rounds
            ],
        }
