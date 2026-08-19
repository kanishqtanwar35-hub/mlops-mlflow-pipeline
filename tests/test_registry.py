"""Promotion-gate tests.

The gate is pure decision logic, so it is tested against fake registry state
rather than a live MLflow server. That means these run in CI in milliseconds
and cover the branches that matter — including the ones that are hard to
produce on demand with real models.
"""

from types import SimpleNamespace

import pytest

from src.mlops.registry import ModelRegistry, PromotionDecision


class FakeRegistry(ModelRegistry):
    """Subclass that replaces every MLflow call with in-memory state."""

    def __init__(self, candidate=None, incumbent=None,
                 candidate_classes=None, incumbent_classes=None):
        self.model_name = "test-model"
        self._candidate = candidate
        self._incumbent = incumbent
        self._candidate_classes = candidate_classes or {}
        self._incumbent_classes = incumbent_classes or {}
        self.promoted = None

    def latest_version(self):
        return self._candidate

    def latest_in_stage(self, stage):
        return self._incumbent

    def metric_for(self, version, metric):
        return version.metrics.get(metric) if version else None

    def per_class_metrics(self, version, prefix):
        if version is self._candidate:
            return self._candidate_classes
        return self._incumbent_classes

    def promote(self, version, archive_existing=True):
        self.promoted = version


def v(version, **metrics):
    return SimpleNamespace(version=str(version), run_id=f"run{version}", metrics=metrics)


def evaluate(reg, **kw):
    defaults = dict(metric="f1", min_absolute=0.60, min_improvement=0.01,
                    per_class_prefix="f1_class_", max_per_class_drop=0.05)
    defaults.update(kw)
    return reg.evaluate_promotion(**defaults)


# -- bar 1: absolute floor ---------------------------------------------------

def test_rejects_when_below_absolute_floor():
    d = evaluate(FakeRegistry(candidate=v(1, f1=0.55)))
    assert d.promote is False
    assert "below the absolute floor" in d.reasons[0]


def test_first_model_promoted_on_the_floor_alone():
    reg = FakeRegistry(candidate=v(1, f1=0.72))
    d = evaluate(reg)
    assert d.promote is True
    assert "no production model yet" in d.reasons[0]


def test_rejects_when_nothing_is_registered():
    d = evaluate(FakeRegistry(candidate=None))
    assert d.promote is False
    assert "no registered versions" in d.reasons[0]


def test_rejects_when_metric_missing():
    d = evaluate(FakeRegistry(candidate=v(1, accuracy=0.9)))
    assert d.promote is False
    assert "no logged metric" in d.reasons[0]


# -- bar 2: improvement over the incumbent -----------------------------------

def test_promotes_on_a_clear_improvement():
    reg = FakeRegistry(candidate=v(2, f1=0.78), incumbent=v(1, f1=0.72))
    d = evaluate(reg)
    assert d.promote is True
    assert d.delta == pytest.approx(0.06)


def test_rejects_a_regression_and_says_so():
    """A drop must not be described as 'within noise' — that would mislead."""
    reg = FakeRegistry(candidate=v(2, f1=0.65), incumbent=v(1, f1=0.73))
    d = evaluate(reg)
    assert d.promote is False
    assert "REGRESSED" in d.reasons[0]
    assert "noise" not in d.reasons[0]


def test_rejects_an_improvement_smaller_than_the_noise_floor():
    reg = FakeRegistry(candidate=v(2, f1=0.725), incumbent=v(1, f1=0.72))
    d = evaluate(reg)
    assert d.promote is False
    assert "within run-to-run noise" in d.reasons[0]


def test_exactly_meeting_the_threshold_promotes():
    reg = FakeRegistry(candidate=v(2, f1=0.73), incumbent=v(1, f1=0.72))
    assert evaluate(reg).promote is True


# -- bar 3: per-class regression ---------------------------------------------

def test_rejects_when_a_class_collapses_despite_a_better_average():
    """The case the whole gate exists for: the headline metric improves while
    the model stops working for one class."""
    reg = FakeRegistry(
        candidate=v(2, f1=0.80), incumbent=v(1, f1=0.72),
        candidate_classes={"f1_class_0": 0.95, "f1_class_1": 0.40},
        incumbent_classes={"f1_class_0": 0.75, "f1_class_1": 0.68},
    )
    d = evaluate(reg)
    assert d.promote is False
    assert "per-class regression" in d.reasons[0]
    assert "f1_class_1" in d.reasons[0]


def test_small_per_class_dip_is_tolerated():
    reg = FakeRegistry(
        candidate=v(2, f1=0.80), incumbent=v(1, f1=0.72),
        candidate_classes={"f1_class_0": 0.85, "f1_class_1": 0.66},
        incumbent_classes={"f1_class_0": 0.75, "f1_class_1": 0.68},
    )
    assert evaluate(reg).promote is True


def test_new_class_absent_from_incumbent_is_ignored():
    reg = FakeRegistry(
        candidate=v(2, f1=0.80), incumbent=v(1, f1=0.72),
        candidate_classes={"f1_class_0": 0.85},
        incumbent_classes={"f1_class_0": 0.80, "f1_class_9": 0.90},
    )
    assert evaluate(reg).promote is True


# -- shape -------------------------------------------------------------------

def test_decision_serialises():
    import json

    reg = FakeRegistry(candidate=v(2, f1=0.78), incumbent=v(1, f1=0.72))
    json.dumps(evaluate(reg).as_dict())


def test_decision_records_both_versions():
    reg = FakeRegistry(candidate=v(7, f1=0.78), incumbent=v(3, f1=0.72))
    d = evaluate(reg)
    assert d.candidate_version == "7"
    assert d.incumbent_version == "3"
