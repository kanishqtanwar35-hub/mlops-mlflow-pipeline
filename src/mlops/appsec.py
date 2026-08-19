"""Application-level security: the supply chain, not the request path.

Request handling (`security.py`) is the shallow layer. The controls here cover
the two ways an ML application actually gets compromised, both of which look
like ordinary lines of code:

  1. **SSRF via data ingestion.** `urlretrieve(config_url)` will fetch
     `file:///etc/passwd`, or `http://169.254.169.254/` — the cloud metadata
     endpoint that hands out IAM credentials to anything on the box. A pipeline
     that reads its source URL from config, and a config that anyone can open a
     PR against, is a credential exfiltration primitive.

  2. **Deserialisation RCE.** `joblib.load()` is pickle underneath, and pickle
     executes arbitrary code on load. Anyone who can write to the artifacts
     directory — a compromised CI runner, a shared volume, a path-traversal bug
     — owns the inference process. "Only load models you trust" is the standard
     advice; this makes "trust" mean something checkable.

Neither is exotic. Both are one line of ordinary-looking code.
"""

import hashlib
import ipaddress
import socket
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

from src.mlops.logger import logger

ALLOWED_SCHEMES = {"http", "https"}

# Loading a model is expensive and unbounded by default; a 5 GB artifact on a
# small container is a denial of service whether or not it is malicious.
MAX_MODEL_BYTES = 512 * 1024 * 1024


class SecurityError(RuntimeError):
    """Raised when a control blocks an operation. Never caught and ignored."""


# ---------------------------------------------------------------------------
# SSRF
# ---------------------------------------------------------------------------

def _is_blocked_ip(ip: ipaddress._BaseAddress) -> Optional[str]:
    """Reject anything that is not a public, routable address.

    `is_link_local` is the important one: 169.254.169.254 is the cloud metadata
    service on AWS, GCP and Azure. Fetching it returns instance credentials.
    """
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local (cloud metadata range)"
    if ip.is_private:
        return "private network"
    if ip.is_reserved:
        return "reserved"
    if ip.is_multicast:
        return "multicast"
    if not ip.is_global:
        return "non-routable"
    return None


def validate_source_url(url: str, allow_private: bool = False) -> str:
    """Check a URL before fetching it. Raises SecurityError if unsafe.

    Resolution happens here, and that is deliberate — a hostname that looks
    innocuous can resolve to 127.0.0.1 or to the metadata IP. Checking the
    string alone catches nothing.

    Known limitation, stated rather than hidden: this is a check-then-use
    pattern, so a DNS entry that changes between this call and the fetch
    (DNS rebinding) would slip through. Closing that properly means resolving
    once and connecting to the resolved IP with the Host header pinned, which
    needs a custom transport adapter. For a pipeline reading a URL from a
    reviewed config file, this is the right level of paranoia; for a service
    fetching user-supplied URLs, it is not enough.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        # file:// is the one people forget. urlretrieve supports it, so a
        # config value of file:///etc/shadow is a working exfiltration path.
        raise SecurityError(
            f"blocked URL scheme {parsed.scheme!r} (allowed: {sorted(ALLOWED_SCHEMES)})"
        )

    if not parsed.hostname:
        raise SecurityError(f"URL has no host: {url!r}")

    if allow_private:
        logger.warning("SSRF checks bypassed for %s — allow_private is set", parsed.hostname)
        return url

    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise SecurityError(f"could not resolve {parsed.hostname!r}: {e}") from e

    for info in infos:
        address = info[4][0]
        ip = ipaddress.ip_address(address)
        reason = _is_blocked_ip(ip)
        if reason:
            raise SecurityError(
                f"{parsed.hostname!r} resolves to {address} ({reason}) — refusing to fetch"
            )

    logger.info("url check passed: %s", parsed.hostname)
    return url


# ---------------------------------------------------------------------------
# Path containment
# ---------------------------------------------------------------------------

def safe_path(candidate, root=None) -> Path:
    """Resolve `candidate` and require it to stay inside `root`.

    Config files are inputs too. A `model_file: ../../../../etc/cron.d/evil`
    turns a pipeline that writes artifacts into a pipeline that writes wherever
    it likes. `Path.resolve()` collapses the `..` segments so the containment
    check sees the real destination.
    """
    root = Path(root or Path.cwd()).resolve()
    resolved = Path(candidate).resolve()

    if root not in resolved.parents and resolved != root:
        raise SecurityError(f"path {resolved} escapes {root}")
    return resolved


# ---------------------------------------------------------------------------
# Model integrity
# ---------------------------------------------------------------------------

def file_digest(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_file(path, expected_sha256: Optional[str] = None,
                      max_bytes: int = MAX_MODEL_BYTES) -> Path:
    """Check a model artifact before handing it to a pickle-based loader.

    This does not make `joblib.load` safe — nothing does, short of a format
    that is not pickle. What it does is make tampering *detectable*: if the file
    on disk is not the file the pipeline produced, the digest changes and the
    load is refused before any bytes reach the unpickler.

    Record the digest at training time, check it at serving time. That turns
    "only load models you trust" from advice into a control.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no model at {path}")

    size = path.stat().st_size
    if size > max_bytes:
        raise SecurityError(
            f"model at {path} is {size} bytes, above the {max_bytes} limit"
        )

    if expected_sha256:
        actual = file_digest(path)
        # Plain != is fine here: the digest is not a secret, so there is no
        # timing channel worth closing, and being explicit about why beats
        # cargo-culting compare_digest everywhere.
        if actual != expected_sha256:
            raise SecurityError(
                f"model integrity check FAILED for {path}\n"
                f"  expected {expected_sha256}\n"
                f"  actual   {actual}\n"
                "The artifact is not the one this pipeline produced. Refusing to "
                "load it — joblib deserialisation executes arbitrary code."
            )
        logger.info("model integrity verified: %s", expected_sha256[:16])
    else:
        logger.warning(
            "loading %s with no expected digest — integrity is unverified", path
        )

    return path


def write_manifest(model_path, manifest_path) -> dict:
    """Record what was produced, so a later load can verify it."""
    import json

    path = Path(model_path)
    manifest = {
        "model_file": path.name,
        "sha256": file_digest(path),
        "size_bytes": path.stat().st_size,
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("wrote integrity manifest %s", manifest_path)
    return manifest


def read_expected_digest(manifest_path) -> Optional[str]:
    import json

    path = Path(manifest_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("sha256")
    except (ValueError, OSError):
        logger.warning("could not read integrity manifest at %s", path)
        return None


def redact(text: str, secrets: Iterable[str]) -> str:
    """Strip known secret values out of a string before it is logged.

    Exception messages routinely contain the thing that caused them, which for
    an auth failure is the credential. Logs are copied into tickets and pasted
    into chats far more casually than credential stores are.
    """
    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "***REDACTED***")
    return text
