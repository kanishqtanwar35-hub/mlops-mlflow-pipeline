"""Model registry and the promotion gate.

Training a better model is the easy half. The hard half is deciding whether it
is safe to replace the one currently serving traffic — and doing that by rule
rather than by whoever is confident in the meeting.

The gate here refuses promotion unless the candidate clears three separate
bars, and it treats a per-class collapse as disqualifying even when the
headline metric improved. That last rule is the one that matters: an average
can rise while the model becomes useless for a minority class.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from src.mlops.logger import logger

STAGING = "Staging"
PRODUCTION = "Production"
ARCHIVED = "Archived"


@dataclass
class PromotionDecision:
    promote: bool
    reasons: List[str] = field(default_factory=list)
    candidate_version: Optional[str] = None
    incumbent_version: Optional[str] = None
    candidate_metric: Optional[float] = None
    incumbent_metric: Optional[float] = None
    delta: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "promote": self.promote,
            "reasons": self.reasons,
            "candidate_version": self.candidate_version,
            "incumbent_version": self.incumbent_version,
            "candidate_metric": self.candidate_metric,
            "incumbent_metric": self.incumbent_metric,
            "delta": self.delta,
        }


class ModelRegistry:
    def __init__(self, model_name: str, tracking_uri: Optional[str] = None):
        # Must resolve to the same backend train.py wrote to, otherwise the
        # registry looks empty and the gate silently promotes nothing.
        uri = (
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI")
            or f"sqlite:///{Path('mlflow.db').resolve().as_posix()}"
        )
        mlflow.set_tracking_uri(uri)
        self.client = MlflowClient()
        self.model_name = model_name

    # -- lookups -------------------------------------------------------------

    def _versions(self) -> List:
        try:
            return list(self.client.search_model_versions(f"name='{self.model_name}'"))
        except Exception:
            return []

    def latest_in_stage(self, stage: str):
        matches = [v for v in self._versions() if v.current_stage == stage]
        if not matches:
            return None
        return max(matches, key=lambda v: int(v.version))

    def latest_version(self):
        versions = self._versions()
        if not versions:
            return None
        return max(versions, key=lambda v: int(v.version))

    def metric_for(self, version, metric: str) -> Optional[float]:
        try:
            run = self.client.get_run(version.run_id)
            value = run.data.metrics.get(metric)
            return float(value) if value is not None else None
        except Exception:
            return None

    def per_class_metrics(self, version, prefix: str) -> Dict[str, float]:
        """Collect every logged metric starting with `prefix`, e.g. `f1_class_`."""
        try:
            run = self.client.get_run(version.run_id)
            return {
                k: float(v) for k, v in run.data.metrics.items() if k.startswith(prefix)
            }
        except Exception:
            return {}

    # -- the gate ------------------------------------------------------------

    def evaluate_promotion(
        self,
        metric: str,
        min_absolute: float,
        min_improvement: float = 0.0,
        per_class_prefix: str = "f1_class_",
        max_per_class_drop: float = 0.05,
    ) -> PromotionDecision:
        """Decide whether the newest version should replace production.

        Three bars, all of which must be cleared:

        1. **Absolute floor.** The candidate must be good enough on its own
           terms. A model that beats a terrible incumbent is still terrible.
        2. **Improvement over the incumbent.** `min_improvement` should be set
           above your run-to-run noise, otherwise you promote on randomness and
           the registry fills with churn.
        3. **No per-class collapse.** The headline metric is an average. A
           candidate can gain overall while losing a class entirely, and that
           is a worse model no matter what the mean says.
        """
        decision = PromotionDecision(promote=False)

        candidate = self.latest_version()
        if candidate is None:
            decision.reasons.append("no registered versions exist")
            return decision

        decision.candidate_version = candidate.version
        candidate_metric = self.metric_for(candidate, metric)
        decision.candidate_metric = candidate_metric

        if candidate_metric is None:
            decision.reasons.append(f"candidate has no logged metric '{metric}'")
            return decision

        # --- bar 1: absolute floor
        if candidate_metric < min_absolute:
            decision.reasons.append(
                f"{metric}={candidate_metric:.4f} is below the absolute floor "
                f"{min_absolute:.4f}"
            )
            return decision

        incumbent = self.latest_in_stage(PRODUCTION)

        # First model ever: only the absolute floor applies.
        if incumbent is None:
            decision.promote = True
            decision.reasons.append(
                f"no production model yet; candidate clears the floor "
                f"({candidate_metric:.4f} >= {min_absolute:.4f})"
            )
            return decision

        decision.incumbent_version = incumbent.version
        incumbent_metric = self.metric_for(incumbent, metric)
        decision.incumbent_metric = incumbent_metric

        if incumbent_metric is None:
            decision.promote = True
            decision.reasons.append(
                "incumbent has no comparable metric; promoting on the floor alone"
            )
            return decision

        delta = candidate_metric - incumbent_metric
        decision.delta = round(delta, 6)

        # --- bar 2: beat the incumbent by more than noise
        if delta < min_improvement:
            # "Worse" and "not enough better" are different findings and the
            # operator needs to know which. Reporting a 7-point regression as
            # "inside run-to-run noise" would be actively misleading.
            if delta < 0:
                decision.reasons.append(
                    f"{metric} REGRESSED by {abs(delta):.4f} "
                    f"({incumbent_metric:.4f} -> {candidate_metric:.4f})"
                )
            else:
                decision.reasons.append(
                    f"{metric} improved by only {delta:+.4f}, below the required "
                    f"{min_improvement:+.4f} — within run-to-run noise, so this "
                    "is not evidence of a better model"
                )
            return decision

        # --- bar 3: no class may regress badly
        cand_classes = self.per_class_metrics(candidate, per_class_prefix)
        inc_classes = self.per_class_metrics(incumbent, per_class_prefix)
        regressions = []
        for key, inc_value in inc_classes.items():
            cand_value = cand_classes.get(key)
            if cand_value is None:
                continue
            drop = inc_value - cand_value
            if drop > max_per_class_drop:
                regressions.append(f"{key}: {inc_value:.3f} -> {cand_value:.3f}")

        if regressions:
            decision.reasons.append(
                "per-class regression beyond "
                f"{max_per_class_drop:.2f}: " + "; ".join(regressions)
            )
            decision.reasons.append(
                "a net gain does not offset a class the model stopped serving"
            )
            return decision

        decision.promote = True
        decision.reasons.append(
            f"{metric} {incumbent_metric:.4f} -> {candidate_metric:.4f} "
            f"({delta:+.4f}), no per-class regression"
        )
        return decision

    # -- transitions ---------------------------------------------------------

    def promote(self, version: str, archive_existing: bool = True) -> None:
        logger.info("promoting version %s to %s", version, PRODUCTION)
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=PRODUCTION,
            archive_existing_versions=archive_existing,
        )

    def stage(self, version: str) -> None:
        self.client.transition_model_version_stage(
            name=self.model_name, version=version, stage=STAGING,
        )
