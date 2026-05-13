"""Cross-judge agreement: same recipe, different seeds, agreement statistics.

Invoked by `judge_experiment(..., n_seeds=N)` when N > 1. For each selected
episode the judge is run N times (varying `seed` if the driver supports it,
otherwise just multiple invocations); the writer here collapses the N
judgments into one row per `(trajectory_id, recipe)` with modal blame +
agreement fraction.

Cross-recipe comparison stays out of scope — well-defined only on shared base
fields, and not the most useful signal in practice.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Sequence

from cube_harness.eval_log import BaseJudgeOutput, JudgeMetadata

logger = logging.getLogger(__name__)

AGREEMENT_REPORT_FILENAME = "cross_judge_agreement.csv"
AGREEMENT_COLUMNS: tuple[str, ...] = (
    "trajectory_id",
    "recipe",
    "n_judgments",
    "seeds",
    "primary_blame_modal",
    "primary_blame_agreement",
    "outcome_modal",
    "outcome_agreement",
    "confidence_mean",
)


def _modal(values: Sequence[str]) -> tuple[str, float]:
    """Return `(modal_value, fraction_in_agreement)`. Empty input returns `("", 0.0)`."""
    if not values:
        return "", 0.0
    counter = Counter(values)
    modal_value, count = counter.most_common(1)[0]
    return modal_value, count / len(values)


def _seed_label(meta: JudgeMetadata) -> str:
    """Best-effort seed label for a judge run.

    `JudgeMetadata` doesn't carry a seed today; we approximate with the
    timestamp suffix so two runs at the same model end up with distinct labels."""
    return f"t{int(meta.timestamp)}"


def write_cross_judge_agreement(
    experiment_dir: Path,
    *,
    judgments: dict[tuple[str, str], list[tuple[BaseJudgeOutput, JudgeMetadata]]],
) -> Path:
    """Write `cross_judge_agreement.csv` for `judgments` under `experiment_dir`.

    `judgments` is keyed by `(trajectory_id, recipe_name)`; the value is the list
    of (output, metadata) pairs from N seeds. Rows with `n_judgments < 2` are
    skipped — single-seed cells have no notion of agreement.

    Atomic write: writes to `<name>.tmp` then renames.
    """
    experiment_dir = Path(experiment_dir)
    out = experiment_dir / AGREEMENT_REPORT_FILENAME
    tmp = out.with_suffix(out.suffix + ".tmp")

    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(AGREEMENT_COLUMNS))
        w.writeheader()
        for (trajectory_id, recipe), runs in sorted(judgments.items()):
            if len(runs) < 2:
                continue
            primary_blames = [o.primary_blame.value for o, _ in runs]
            outcomes = [o.outcome.value for o, _ in runs]
            confidences = [o.primary_blame_confidence for o, _ in runs]
            seeds = ";".join(_seed_label(m) for _, m in runs)
            blame_modal, blame_agree = _modal(primary_blames)
            outcome_modal, outcome_agree = _modal(outcomes)
            w.writerow(
                {
                    "trajectory_id": trajectory_id,
                    "recipe": recipe,
                    "n_judgments": len(runs),
                    "seeds": seeds,
                    "primary_blame_modal": blame_modal,
                    "primary_blame_agreement": round(blame_agree, 4),
                    "outcome_modal": outcome_modal,
                    "outcome_agreement": round(outcome_agree, 4),
                    "confidence_mean": round(sum(confidences) / len(confidences), 4),
                }
            )
    tmp.replace(out)
    logger.info("Wrote %s (%d (trajectory, recipe) cells)", out, sum(1 for v in judgments.values() if len(v) >= 2))
    return out


__all__ = [
    "AGREEMENT_REPORT_FILENAME",
    "AGREEMENT_COLUMNS",
    "write_cross_judge_agreement",
]
