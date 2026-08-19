# MLOps Pipeline — MLflow tracking, model registry, promotion gate, CI/CD

Every training run is tracked. Every model is versioned. Nothing reaches
production without passing a written policy.

The interesting part is not the tracking — it is the **promotion gate**: three
rules that decide whether a new model is allowed to replace the one currently
serving traffic, evaluated automatically in CI.

**Status:** verified running. Two models trained and registered, v1 promoted,
v2 correctly rejected, 13/13 tests pass.

---

## What actually happened when this was built

```
$ python train.py
  f1=0.7275  accuracy=0.7597  registered=churn-risk-classifier
  Created version '1'

$ python promote.py
  candidate    v1  f1=0.7275
  production   none
    - no production model yet; candidate clears the floor (0.7275 >= 0.6000)
  DECISION: PROMOTED v1 to Production                              exit 0

$ python train.py --run-name underfit-depth2 --set max_depth=2 --set n_estimators=10
  f1=0.6562  accuracy=0.7273
  Created version '2'

$ python promote.py
  candidate    v2  f1=0.6562
  production   v1  f1=0.7275
  delta        -0.0712
    - f1 REGRESSED by 0.0712 (0.7275 -> 0.6562)
  DECISION: REJECTED                                                exit 1
```

The worse model was registered — you always want the record — and then refused
promotion. Production stayed on v1. In CI that non-zero exit turns the build
red.

---

## The promotion gate

Three bars. A candidate must clear **all three**.

### 1. Absolute floor
```yaml
min_absolute: 0.60
```
The candidate has to be good on its own terms. A model that beats a terrible
incumbent is still terrible.

### 2. Improvement above noise
```yaml
min_improvement: 0.01
```
Set this from the standard deviation of repeated runs, not by taste. Set it to
`0.0` and you promote on randomness — the registry fills with churn and the
version history stops meaning anything.

The gate distinguishes *worse* from *not enough better*, because those are
different findings:

- `f1 REGRESSED by 0.0712 (0.7275 -> 0.6562)`
- `f1 improved by only +0.0040, below the required +0.0100 — within run-to-run noise`

An earlier version reported the first case using the second message. Describing
a seven-point regression as "within noise" is exactly the kind of small
inaccuracy that erodes trust in a system nobody reads closely.

### 3. No per-class collapse
```yaml
per_class_prefix: f1_class_
max_per_class_drop: 0.05
```
**This is the bar that justifies the whole project.** The headline metric is an
average, and an average can rise while the model stops working for a class
entirely:

| | v1 (production) | v2 (candidate) |
|---|---|---|
| macro f1 | 0.72 | **0.80** ↑ |
| f1_class_0 | 0.75 | 0.95 |
| f1_class_1 | 0.68 | **0.40** ↓ |

v2 looks better and is worse. Any gate reading only the headline promotes it.
`test_rejects_when_a_class_collapses_despite_a_better_average` pins the
behaviour.

---

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python train.py                 # train, log, register a version
python promote.py               # evaluate the policy, promote or reject
pytest -q                       # 13 tests
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open <http://localhost:5000>.

### Running a sweep

```bash
for d in 4 8 12 16; do python train.py --run-name "depth-$d" --set max_depth=$d; done
python promote.py
```

Overrides exist so a sweep is a shell loop rather than fifteen edited copies of
`params.yaml` — and every variant still lands in MLflow with its exact
parameters attached.

---

## What each run records

| Recorded | Why |
|---|---|
| All hyperparameters | Obvious, and still the thing most ad-hoc logs get wrong |
| Headline metrics | accuracy, f1, precision, recall, roc_auc |
| **Per-class F1**, as separate metrics | So the gate can see a single class collapsing |
| **`data_sha256`** | Two runs are not comparable if the data moved underneath them, and that failure is invisible unless something records it |
| **`git_commit` tag** | Without it a run is a set of numbers with no route back to the code that produced them |

---

## Design decisions worth defending

**SQLite, not the `./mlruns` file store.** MLflow 3.x put the file backend into
maintenance mode and refuses it by default, and the model registry has never
worked properly on it — stage transitions need a real database. A local `.db`
keeps this offline-capable and CI-friendly while behaving like a real server.
Point `MLFLOW_TRACKING_URI` at a hosted server and nothing else changes.

**Registering and promoting are separate commands.** `train.py` always
registers; only `promote.py` decides what serves traffic. Collapsing them means
every training run is a deploy.

**skops with an explicit trust list, not cloudpickle.** MLflow 3.x serialises
sklearn models with skops and refuses types not on a trust list. Loading a
pickled model is arbitrary code execution, so the right response is to declare
what the pipeline actually contains rather than switching back to pickle to
make the error go away.

**Nothing in config identifies an account.** No tracking URL, no username, no
registry ID. Hardcode a DagsHub URL and every fork of the repo posts its
metrics to your account — a real bug found in widely-copied tutorial code.

**The gate is tested against fake registry state.** The policy is pure decision
logic, so its tests need no MLflow server and run in milliseconds — including
the branches that are hard to produce on demand with real models, like a class
collapsing while the average improves.

---

## CI

`.github/workflows/mlops.yml` runs on push, on PRs, weekly, and on manual
dispatch with a model name.

On a **pull request** the gate runs `--dry-run`: the candidate is evaluated and
reported, production is untouched. On **main** it runs for real.

The weekly schedule matters more than it looks. A pipeline that only runs when
someone remembers to run it is not a pipeline, and the scheduled run also
catches dependency drift breaking training before it matters.

---

## Limitations, stated plainly

- Local SQLite backend. Fine for one machine, wrong for a team — point
  `MLFLOW_TRACKING_URI` at a shared server for that.
- No serving component. The registry knows which version is production; a
  service that loads it is a separate concern.
- The demonstration dataset is small (768 rows) and the metric variance is
  correspondingly high. `min_improvement: 0.01` is a plausible noise floor for
  it, not a measured one — measure yours.
- No drift detection or automatic rollback. Both are natural next steps.
