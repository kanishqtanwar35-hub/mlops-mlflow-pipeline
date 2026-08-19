"""Shared helpers."""

import json
from pathlib import Path
from typing import Any, List

import yaml
from box import ConfigBox


def read_yaml(path: Any) -> ConfigBox:
    with open(Path(path), "r", encoding="utf-8") as f:
        return ConfigBox(yaml.safe_load(f))


def create_directories(paths: List[Any]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_json(path: Any, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Any) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)
