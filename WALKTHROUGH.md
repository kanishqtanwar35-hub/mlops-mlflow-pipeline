# Code walkthrough — MLOps pipeline

```
train.py                                promote.py
   |                                        |
   +-- tracking.ExperimentTracker           +-- registry.ModelRegistry
   |     params, metrics, artifacts         |     .evaluate_promotion()
   |     git_commit, data_sha256            |        bar 1  absolute floor
   |                                        |        bar 2  beats incumbent
   +-- training.build_pipeline / evaluate   |        bar 3  no class collapse
   |                                        |
   +-- log_model(registered_name=...)       +-- .promote()  or  exit 1
         -> registry version N                    -> Production
```

Two commands, deliberately separate. Training always registers a version;
only promotion decides what serves traffic. Collapse them and every training
run becomes a deploy.

---

## `tracking.py` — what a run has to record

The dashboard is not the point. The point is that six weeks later you can
answer "what produced this model?" without guessing. Three fields do most of
that work:

**`git_commit`** — a run without it is a set of numbers with no route back to
the source that made them. This is the single most common gap in ad-hoc
experiment logs.

**`data_sha256`** — two runs are not comparable if the data moved underneath
them, and that failure is completely invisible unless something records what
the data was. Hashing in 1 MB chunks keeps it cheap on large files.

**Flattened params** — MLflow rejects nested structures. `log_params` flattens
one level (`model.max_depth`) rather than silently dropping the interesting
half of the config, which is what passing the raw dict would do.

### The skops trust list

```python
TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.compose._column_transformer._RemainderColsList",
]
```

MLflow 3.x serialises sklearn with skops instead of pickle and refuses any type
not on an explicit list. The first run failed on exactly this.

There were two ways out: pass `serialization_format="cloudpickle"` and go back
to pickle, or declare what the pipeline actually contains. Loading a pickled
model is arbitrary code execution — an attacker who can write to your artifact
store owns your inference box — so the trust list is the right answer even
though it is more work. The error was a feature.

## `registry.py` — the gate

`evaluate_promotion` returns a `PromotionDecision`, never mutates anything, and
`promote.py` decides what to do with it. Separating the decision from the action
is what makes the policy testable without a live server.

### Why three bars and not one

**Bar 1, absolute floor.** Beating a terrible incumbent still leaves you
terrible. Without this, one bad deploy sets a low bar that every subsequent
model clears.

**Bar 2, improvement above noise.** `min_improvement` should come from the
standard deviation of repeated runs. At `0.0` you promote on randomness and the
version history stops carrying information.

The message distinguishes two cases that are genuinely different:

```python
if delta < 0:
    "f1 REGRESSED by 0.0712 (0.7275 -> 0.6562)"
else:
    "f1 improved by only +0.0040 ... within run-to-run noise"
```

The first version of this code used the second message for both. Reporting a
seven-point regression as "within noise" is a small inaccuracy that quietly
teaches people to ignore the output.

**Bar 3, per-class collapse.** The one that justifies the project:

| | incumbent | candidate |
|---|---|---|
| macro f1 | 0.72 | **0.80** ↑ |
| f1_class_1 | 0.68 | **0.40** ↓ |

The candidate's average improved by eight points while it stopped working for
one class. A gate reading only the headline promotes it. This is why
`training.evaluate` logs per-class F1 as *separate metrics* — `f1_class_0`,
`f1_class_1` — rather than as one nested blob: MLflow can only compare scalars,
so anything the gate needs to reason about has to be logged as its own metric.

### The uri resolution bug worth noticing

```python
uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{...}"
mlflow.set_tracking_uri(uri)
```

`ModelRegistry` resolves the backend the same way `ExperimentTracker` does. If
it did not, the registry would connect to a *different* store, find it empty,
and the gate would report "no registered versions" — silently promoting nothing
while looking like it ran correctly. Two components that must agree on a
location should compute it the same way, not each guess.

## `training.py` — small, and deliberately boring

One preprocessor, one estimator, one `Pipeline`. The model is saved as a single
object so serving cannot drift from training.

`evaluate` returns headline metrics *and* one entry per class. That shape is
driven entirely by what the gate needs to see.

## `promote.py` — the exit code

```python
if not decision.promote:
    return 1
```

A report nobody reads changes nothing; a build that goes red changes behaviour.
`--dry-run` exists so a pull request can be evaluated against the policy without
altering production, which is how the CI workflow uses it.

## `tests/test_registry.py` — testing a policy without a server

`FakeRegistry` subclasses the real one and replaces every MLflow call with
in-memory state. Thirteen tests, milliseconds, no server, no network.

That design is what makes it possible to test the branches that matter most and
are hardest to produce on demand: a class collapsing while the average improves,
a first-ever model with no incumbent, a candidate missing the metric entirely.
Waiting for those to occur naturally with real models is not a test strategy.

---

## What to add next

1. **Drift detection** — compare live input distributions against the training
   fingerprint, and trigger retraining when they separate.
2. **Automatic rollback** — if the promoted model's live metrics degrade,
   transition the previous version back to Production.
3. **A serving component** that reads `Production` from the registry at startup,
   so deploys follow promotions without a code change.
4. **A shared tracking server** — set `MLFLOW_TRACKING_URI` and nothing in this
   repository changes.
