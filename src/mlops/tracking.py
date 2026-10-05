"""MLflow experiment tracking.

Everything about a run that you would otherwise have to remember: the params,
the metrics, the artifacts, the git commit, the data fingerprint. The point is
not the dashboard — it is that six weeks later you can answer "what produced
this model?" without guessing.

The tracking URI defaults to a local `mlruns/` directory so this works offline
and in CI with no server. Point `MLFLOW_TRACKING_URI` at a real server (or a
DagsHub URL) and nothing else changes.
"""

import hashlib
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow

from src.mlops.logger import logger


def git_commit() -> str:
    """Record which code produced the model.

    Without this, a run is a set of numbers with no way back to the source that
    made them — which is the single most common gap in ad-hoc experiment logs.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def data_fingerprint(path: Path) -> str:
    """Hash the input data so a silent dataset change is visible in the diff.

    Comparing two runs is meaningless if the data moved underneath them, and
    that failure is invisible unless something records what the data was.
    """
    path = Path(path)
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


class ExperimentTracker:
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        # SQLite rather than the old `./mlruns` file store. Two reasons: MLflow
        # 3.x put the file backend into maintenance mode and refuses it by
        # default, and the model registry has never worked properly on it —
        # stage transitions need a real database. A local .db file keeps this
        # offline-capable and CI-friendly while behaving like a real server.
        self.tracking_uri = (
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI")
            or f"sqlite:///{Path('mlflow.db').resolve().as_posix()}"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info("tracking to %s (experiment=%s)", self.tracking_uri, experiment_name)

    @contextmanager
    def run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        base_tags = {"git_commit": git_commit()}
        base_tags.update(tags or {})

        with mlflow.start_run(run_name=run_name, tags=base_tags) as active:
            logger.info("run %s started (id=%s)", run_name, active.info.run_id)
            yield active
            logger.info("run %s finished", run_name)

    @staticmethod
    def log_params(params: Dict[str, Any]) -> None:
        # MLflow rejects nested structures, so flatten one level rather than
        # silently dropping the interesting half of the config.
        flat: Dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, dict):
                for sub, sub_value in value.items():
                    flat[f"{key}.{sub}"] = sub_value
            elif isinstance(value, (list, tuple)):
                flat[key] = ",".join(str(v) for v in value)
            else:
                flat[key] = value
        mlflow.log_params(flat)

    @staticmethod
    def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
        numeric = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        mlflow.log_metrics(numeric, step=step)

    @staticmethod
    def log_artifact(path: Any, artifact_path: Optional[str] = None) -> None:
        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path)

    # MLflow 3.x serialises sklearn models with skops instead of pickle, and
    # refuses to write any type not on an explicit trust list. That is a real
    # security improvement — loading a pickled model is arbitrary code
    # execution — so the right response is to declare what this pipeline
    # actually contains rather than falling back to cloudpickle to make the
    # error go away.
    #
    # Newer skops releases (0.16+) also stopped trusting sklearn.tree._tree.Tree
    # by default. Every tree model here (RandomForest, GradientBoosting) stores
    # its nodes in that type, so without it log_model fails. We only ever load
    # models this pipeline trained itself, so it is safe to trust here.
    TRUSTED_TYPES = [
        "numpy.dtype",
        "sklearn.compose._column_transformer._RemainderColsList",
        "sklearn.tree._tree.Tree",
    ]

    @classmethod
    def log_model(cls, model, artifact_path: str = "model",
                  registered_name: Optional[str] = None):
        import mlflow.sklearn

        return mlflow.sklearn.log_model(
            sk_model=model,
            name=artifact_path,
            registered_model_name=registered_name,
            skops_trusted_types=cls.TRUSTED_TYPES,
        )
