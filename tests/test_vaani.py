import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "data"))
import vaani  # noqa: E402


class _Ok:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def read(self):
        return b"ok"


def _serve(monkeypatch, *events):
    """Make each request answer with the next event: "ok", "reset" (dropped connection) or an HTTP status."""
    calls = []

    def urlopen(req, timeout):
        e = events[len(calls)]
        calls.append(e)
        if e == "ok":
            return _Ok()
        if e == "reset":
            raise ConnectionResetError("reset")
        raise urllib.error.HTTPError(req.full_url, e, "x", {}, None)

    monkeypatch.setattr(vaani.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(vaani.time, "sleep", lambda s: None)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "x")
    return calls


def test_get_retries_server_errors_and_dropped_connections(monkeypatch):
    calls = _serve(monkeypatch, 503, "reset", 429, "ok")
    assert vaani._get("repo", "f", 0, 10) == b"ok" and len(calls) == 4


def test_get_gives_up(monkeypatch):
    calls = _serve(monkeypatch, *[503] * vaani.TRIES)
    with pytest.raises(urllib.error.HTTPError):
        vaani._get("repo", "f", 0, 10)
    assert len(calls) == vaani.TRIES

    calls = _serve(monkeypatch, 404)
    with pytest.raises(urllib.error.HTTPError):
        vaani._get("repo", "f", 0, 10)
    assert len(calls) == 1  # a missing file isn't retried
