"""Run one tracked training experiment and register the model.

    python train.py
    python train.py --model LogisticRegression --set n_estimators=400
    python train.py --run-name "deeper-trees" --set max_depth=16

Everything about the run is logged to MLflow: params, headline and per-class
metrics, the data fingerprint, and the git commit. The model is registered as a
new version — registering is not promoting, which is `promote.py`'s job.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.mlops.logger import logger
from src.mlops.tracking import ExperimentTracker, data_fingerprint
from src.mlops.training import build_pipeline, ensure_data, evaluate, split
from src.mlops.utils import create_directories, read_yaml, save_json


def parse_overrides(pairs) -> dict:
    """`--set max_depth=16 --set n_estimators=400`

    Overrides exist so a sweep is a shell loop rather than fifteen edited
    copies of params.yaml — and every variant still lands in MLflow with its
    exact parameters attached.
    """
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got '{pair}'")
        key, raw = pair.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out[key.strip()] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tracked training experiment.")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--model", default=None, help="override model.name")
    parser.add_argument("--set", action="append", dest="overrides",
                        help="override a model hyperparameter, e.g. max_depth=16")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-register", action="store_true",
                        help="log the run but do not create a registry version")
    args = parser.parse_args()

    config = read_yaml(args.config)
    params = read_yaml(args.params)
    create_directories([config.artifacts.root_dir])

    model_name = args.model or params.model.name
    model_params = dict(params.model.params)
    model_params.update(parse_overrides(args.overrides))

    data_path = ensure_data(
        config.data.source_url,
        Path(config.data.local_file),
        list(config.data.column_names or []),
    )
    df = pd.read_csv(data_path)
    target = config.data.target_column

    X_train, X_test, y_train, y_test = split(
        df, target,
        test_size=float(params.split.test_size),
        seed=int(params.split.random_state),
    )
    class_labels = sorted(df[target].unique().tolist())

    tracker = ExperimentTracker(config.experiment_name)
    run_name = args.run_name or f"{model_name}"

    with tracker.run(run_name, tags={"model": model_name}):
        tracker.log_params({
            "model_name": model_name,
            "model": model_params,
            "test_size": float(params.split.test_size),
            "random_state": int(params.split.random_state),
            "n_train": len(X_train),
            "n_test": len(X_test),
            # Two runs are not comparable if the data moved underneath them,
            # and that failure is invisible unless something records it.
            "data_sha256": data_fingerprint(data_path),
        })

        pipeline = build_pipeline(model_name, model_params, X_train.columns)
        logger.info("fitting %s on %d rows", model_name, len(X_train))
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test, class_labels)
        tracker.log_metrics(metrics)

        save_json(config.artifacts.metrics_file, metrics)
        tracker.log_artifact(config.artifacts.metrics_file)

        registered = None if args.no_register else config.registry.model_name
        tracker.log_model(pipeline, registered_name=registered)

        logger.info("f1=%.4f  accuracy=%.4f  registered=%s",
                    metrics["f1"], metrics["accuracy"], registered or "no")

    print(json.dumps(
        {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
