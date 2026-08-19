"""Application-level security tests: SSRF, path containment, model integrity.

These cover the two ways an ML application actually gets compromised — a config
value that points somewhere it shouldn't, and a tampered artifact reaching a
pickle-based loader.
"""

import json

import pytest

from src.mlops.appsec import (
    SecurityError,
    file_digest,
    read_expected_digest,
    redact,
    safe_path,
    validate_source_url,
    verify_model_file,
    write_manifest,
)


# -- SSRF: schemes -----------------------------------------------------------

def test_file_scheme_blocked():
    """urlretrieve honours file://, so a config value of file:///etc/shadow is
    a working exfiltration path."""
    with pytest.raises(SecurityError, match="scheme"):
        validate_source_url("file:///etc/passwd")


def test_ftp_scheme_blocked():
    with pytest.raises(SecurityError, match="scheme"):
        validate_source_url("ftp://example.com/data.csv")


def test_gopher_scheme_blocked():
    with pytest.raises(SecurityError, match="scheme"):
        validate_source_url("gopher://example.com/")


def test_url_without_host_blocked():
    with pytest.raises(SecurityError):
        validate_source_url("http://")


# -- SSRF: addresses ---------------------------------------------------------

def test_localhost_blocked():
    with pytest.raises(SecurityError, match="loopback"):
        validate_source_url("http://localhost/data.csv")


def test_loopback_ip_blocked():
    with pytest.raises(SecurityError, match="loopback"):
        validate_source_url("http://127.0.0.1/data.csv")


def test_cloud_metadata_endpoint_blocked():
    """169.254.169.254 returns IAM credentials on AWS, GCP and Azure. This is
    the single most valuable SSRF target in any cloud deployment."""
    with pytest.raises(SecurityError, match="link-local|metadata"):
        validate_source_url("http://169.254.169.254/latest/meta-data/")


def test_private_range_blocked():
    with pytest.raises(SecurityError, match="private"):
        validate_source_url("http://10.0.0.5/internal.csv")


def test_another_private_range_blocked():
    with pytest.raises(SecurityError, match="private"):
        validate_source_url("http://192.168.1.1/admin")


def test_unresolvable_host_blocked():
    with pytest.raises(SecurityError, match="resolve"):
        validate_source_url("https://this-host-does-not-exist-8f3a.invalid/x.csv")


def test_allow_private_escape_hatch_works():
    """Explicit opt-in for genuinely internal sources, logged loudly."""
    assert validate_source_url("http://127.0.0.1/x.csv", allow_private=True)


# -- path containment --------------------------------------------------------

def test_path_inside_root_allowed(tmp_path):
    target = tmp_path / "artifacts" / "model.joblib"
    assert safe_path(target, root=tmp_path) == target.resolve()


def test_traversal_outside_root_blocked(tmp_path):
    """A config value of ../../../../etc/cron.d/evil turns a pipeline that
    writes artifacts into a pipeline that writes anywhere."""
    with pytest.raises(SecurityError, match="escapes"):
        safe_path(tmp_path / ".." / ".." / "etc" / "passwd", root=tmp_path)


def test_absolute_path_outside_root_blocked(tmp_path):
    with pytest.raises(SecurityError, match="escapes"):
        safe_path("/etc/passwd", root=tmp_path)


# -- model integrity ---------------------------------------------------------

@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"pretend this is a pickled sklearn pipeline")
    return path


def test_manifest_records_the_digest(model_file, tmp_path):
    manifest = write_manifest(model_file, tmp_path / "model.integrity.json")
    assert manifest["sha256"] == file_digest(model_file)
    assert manifest["size_bytes"] == model_file.stat().st_size


def test_unmodified_model_verifies(model_file, tmp_path):
    manifest_path = tmp_path / "model.integrity.json"
    write_manifest(model_file, manifest_path)
    verify_model_file(model_file, read_expected_digest(manifest_path))


def test_tampered_model_is_refused(model_file, tmp_path):
    """The control that matters: swap the artifact, the load is refused before
    any bytes reach the unpickler."""
    manifest_path = tmp_path / "model.integrity.json"
    write_manifest(model_file, manifest_path)

    model_file.write_bytes(b"malicious pickle payload")

    with pytest.raises(SecurityError, match="integrity check FAILED"):
        verify_model_file(model_file, read_expected_digest(manifest_path))


def test_oversized_model_refused(model_file):
    with pytest.raises(SecurityError, match="above the"):
        verify_model_file(model_file, max_bytes=4)


def test_missing_model_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_model_file(tmp_path / "nope.joblib")


def test_missing_manifest_returns_none(tmp_path):
    assert read_expected_digest(tmp_path / "absent.json") is None


def test_corrupt_manifest_returns_none(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert read_expected_digest(bad) is None


def test_load_without_digest_is_allowed_but_warns(model_file):
    """Unverified load still works — otherwise a fresh clone could never start
    — but it is logged as unverified rather than passing silently."""
    assert verify_model_file(model_file) == model_file


# -- log redaction -----------------------------------------------------------

def test_secrets_redacted_from_text():
    msg = "auth failed for key sk-abcdefghijklmnop"
    assert "sk-abcdefghijklmnop" not in redact(msg, ["sk-abcdefghijklmnop"])
    assert "***REDACTED***" in redact(msg, ["sk-abcdefghijklmnop"])


def test_short_strings_not_redacted():
    """Redacting short values would mangle ordinary log lines for no benefit."""
    assert redact("status is ok", ["ok"]) == "status is ok"


def test_redaction_handles_empty_secret_list():
    assert redact("nothing to hide", []) == "nothing to hide"
