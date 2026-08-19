"""Decide whether the newest registered version should serve production.

    python promote.py            # evaluate and promote if it passes
    python promote.py --dry-run  # evaluate and report, change nothing

Exits 1 when the candidate is rejected, so CI can run this as a gate: a model
that fails the policy turns the build red instead of quietly shipping.

Registering a model and promoting it are deliberately separate steps. `train.py`
always registers; only this decides what serves traffic.
"""

import argparse
import json
import sys
from pathlib import Path

from src.mlops.logger import logger
from src.mlops.registry import ModelRegistry
from src.mlops.utils import create_directories, read_yaml, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the promotion gate.")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate the policy without transitioning anything")
    args = parser.parse_args()

    config = read_yaml(args.config)
    params = read_yaml(args.params)
    create_directories([config.artifacts.root_dir])

    policy = params.promotion
    registry = ModelRegistry(config.registry.model_name)

    decision = registry.evaluate_promotion(
        metric=str(policy.metric),
        min_absolute=float(policy.min_absolute),
        min_improvement=float(policy.min_improvement),
        per_class_prefix=str(policy.per_class_prefix),
        max_per_class_drop=float(policy.max_per_class_drop),
    )

    save_json(config.artifacts.decision_file, decision.as_dict())

    print()
    print(f"model        {config.registry.model_name}")
    print(f"candidate    v{decision.candidate_version}  "
          f"{policy.metric}={decision.candidate_metric}")
    if decision.incumbent_version:
        print(f"production   v{decision.incumbent_version}  "
              f"{policy.metric}={decision.incumbent_metric}")
        print(f"delta        {decision.delta:+.4f}"
              if decision.delta is not None else "delta        n/a")
    else:
        print("production   none")
    print()
    for reason in decision.reasons:
        print(f"  - {reason}")
    print()

    if not decision.promote:
        logger.info("REJECTED — production unchanged")
        print("DECISION: REJECTED")
        return 1

    if args.dry_run:
        logger.info("would promote v%s (dry run)", decision.candidate_version)
        print("DECISION: WOULD PROMOTE (dry run, nothing changed)")
        return 0

    registry.promote(decision.candidate_version)
    logger.info("promoted v%s to Production", decision.candidate_version)
    print(f"DECISION: PROMOTED v{decision.candidate_version} to Production")
    return 0


if __name__ == "__main__":
    sys.exit(main())
