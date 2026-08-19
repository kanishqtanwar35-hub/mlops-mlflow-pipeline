# Security

This project has no HTTP surface, so the request-level controls in the serving
projects do not apply. Its exposure is entirely **supply chain**: what it
downloads, what it deserialises, and what it records.

Implemented in [`src/mlops/appsec.py`](src/mlops/appsec.py), covered by
[`tests/test_appsec.py`](tests/test_appsec.py) (21 tests, no MLflow server
needed).

---

## 1. SSRF in data ingestion

**The line of code.**

```python
urllib.request.urlretrieve(config.data.source_url, local_path)
```

Ordinary-looking. Also a server-side request forgery primitive, because
`urlretrieve` fetches whatever it is handed:

| Payload | Result |
|---|---|
| `file:///etc/passwd` | Reads local files — `urlretrieve` supports `file://` |
| `http://169.254.169.254/latest/meta-data/iam/security-credentials/` | **Returns your IAM credentials.** Same address on AWS, GCP and Azure |
| `http://10.0.0.5/admin` | Reaches internal services |

The second row is why this matters. The cloud metadata endpoint is
unauthenticated and reachable from any process on the instance, and it hands out
role credentials. A training job that reads its source URL from config — a
config anyone can open a pull request against — becomes a credential
exfiltration tool.

**The control.** `validate_source_url()` runs before every fetch: scheme must be
`http`/`https`, and the hostname is **resolved** with every returned address
checked against loopback, link-local, private, reserved and non-global ranges.

**The subtlety:** checking the *string* catches nothing. A perfectly ordinary
hostname can resolve to `127.0.0.1` or `169.254.169.254`. You have to resolve it.

**Stated limitation:** check-then-use, so DNS rebinding would slip through.
Closing that needs a pinned-IP transport adapter. For a pipeline reading a URL
from a reviewed config file this is the right level; for a service fetching
user-supplied URLs it is not.

## 2. Model serialisation — skops, not pickle

MLflow 3.x serialises sklearn models with **skops** and refuses any type not on
an explicit trust list. The first run here failed on exactly that.

There were two ways out:

```python
serialization_format="cloudpickle"        # make the error go away
skops_trusted_types=[...]                 # declare what the pipeline contains
```

The second is correct. **Loading a pickled model is arbitrary code execution** —
an attacker who can write to your artifact store owns your inference process the
instant a model is loaded, before `.predict()` is ever called. The error was a
feature, and the trust list is the control:

```python
TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.compose._column_transformer._RemainderColsList",
]
```

This is the one place in the portfolio where the *format itself* is safe rather
than merely verified. The other projects use joblib and compensate with digest
verification — detection instead of prevention. Prevention is better when you
can have it.

## 3. No account identifiers in configuration

```python
self.tracking_uri = (
    tracking_uri
    or os.getenv("MLFLOW_TRACKING_URI")
    or f"sqlite:///{Path('mlflow.db').resolve().as_posix()}"
)
```

Nothing in `config/` names a server, a username or a registry. This is not
cosmetic: a hardcoded tracking URL in a public repository means **every fork
posts its metrics to your account**. It is extremely common in tutorial code —
including in the reference material this portfolio was built alongside, where a
DagsHub URL for the instructor's personal account is baked into the source.

## 4. Path containment

`safe_path()` resolves a path and requires it to stay inside the project root.
Config files are inputs too, and a `model_file: ../../../../etc/cron.d/evil`
turns an artifact writer into an arbitrary file writer.

## 5. Log redaction

`redact()` strips known secret values before text is logged. Exception messages
routinely contain the value that caused them, which for an auth failure is the
credential — and logs get pasted into tickets far more casually than credential
stores get opened.

## 6. Supply chain

- **Pinned dependencies**, so a compromised or broken upstream release cannot
  silently enter a build.
- **`gitleaks` in CI over full history** — a secret deleted in the latest commit
  is still in the repository and still compromised.
- **`pip-audit`** against the OSV database, advisory rather than blocking: a
  transitive advisory with no available fix should not stop a release, but it
  should be visible.
- **GitHub push protection enabled**, so a detected secret is blocked at push
  time rather than cleaned up afterwards.
- **`mlflow.db` is gitignored.** It is machine-specific state, it grows without
  bound, and a merge conflict inside it is unresolvable.

---

## Deliberately not done, and why

- **No authentication on the tracking server.** There is no server — the default
  backend is a local SQLite file. A shared deployment needs auth, and that
  belongs to the server, not this repository.
- **No artifact signing.** Digest verification proves an artifact is unchanged;
  signing would prove *who* produced it. Worth adding when more than one person
  can publish.
- **No network egress restrictions.** The SSRF control covers the one place this
  code fetches. A hardened deployment restricts egress at the network layer too,
  which is the more robust control.

## Checklist for your own pipelines

1. Validate every URL before fetching it, and **resolve the hostname** — the
   string tells you nothing.
2. Prefer a non-executable model format (skops, ONNX, safetensors). If you must
   use pickle, verify a digest recorded at training time.
3. Keep account identifiers out of config. Environment variables with documented
   defaults.
4. Resolve config-supplied paths and require containment.
5. Scan history for secrets, not just the tip.
6. Write down what you did **not** protect against.
