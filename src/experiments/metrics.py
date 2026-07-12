"""Data structures and CSV writers for the multi-method comparison study."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional

from ..solver import StepStats

# Fixed superset of method-specific ``extra`` keys so the history CSV has stable
# columns regardless of which method produced the row.
EXTRA_KEYS = [
    "species",
    "nodes",
    "connections",
    "epsilon",
    "q_coverage",
    "nodes_expanded",
]


@dataclass
class TrialResult:
    algorithm: str
    seed: int
    solved: bool
    iterations_to_solve: Optional[int]      # native units (generations / steps)
    evaluations_to_solve: Optional[int]     # environment episodes consumed
    wall_time_s: float
    best_path_length: Optional[int]
    optimal_path_length: int
    history: List[StepStats] = field(default_factory=list)

    @property
    def optimality_ratio(self) -> Optional[float]:
        """best_path_length / optimal (1.0 == perfectly optimal)."""
        if self.best_path_length is None or self.optimal_path_length <= 0:
            return None
        return self.best_path_length / self.optimal_path_length


def write_history_csv(trials: List[TrialResult], path: str) -> None:
    """One row per (algorithm, seed, iteration) for convergence plots."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "seed",
                "iteration",
                "evaluations",
                "reached",
                "best_path_length",
                "best_fitness",
            ]
            + EXTRA_KEYS
        )
        for trial in trials:
            for st in trial.history:
                writer.writerow(
                    [
                        trial.algorithm,
                        trial.seed,
                        st.iteration,
                        st.evaluations,
                        int(st.reached),
                        st.best_path_length if st.best_path_length is not None else "",
                        f"{st.best_fitness:.4f}",
                    ]
                    + [st.extra.get(k, "") for k in EXTRA_KEYS]
                )


def write_summary_csv(trials: List[TrialResult], path: str) -> None:
    """One row per trial summarizing the outcome."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "seed",
                "solved",
                "iterations_to_solve",
                "evaluations_to_solve",
                "wall_time_s",
                "best_path_length",
                "optimal_path_length",
                "optimality_ratio",
            ]
        )
        for t in trials:
            writer.writerow(
                [
                    t.algorithm,
                    t.seed,
                    int(t.solved),
                    t.iterations_to_solve if t.iterations_to_solve is not None else "",
                    t.evaluations_to_solve if t.evaluations_to_solve is not None else "",
                    f"{t.wall_time_s:.4f}",
                    t.best_path_length if t.best_path_length is not None else "",
                    t.optimal_path_length,
                    f"{t.optimality_ratio:.4f}" if t.optimality_ratio else "",
                ]
            )
