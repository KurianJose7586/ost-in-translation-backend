import re
import unicodedata
from pathlib import Path

from atlas_eval.normalize import lenient, normalise


def unescape(s):
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), s)


def test_pairs():
    lines = Path(__file__).with_name("normalize_pairs.tsv").read_text(encoding="utf-8").splitlines()[1:]
    rows = [line.split("\t") for line in lines]
    assert len(rows) >= 50
    failures = []
    for inp, want, rule in rows:
        got, want = normalise(unescape(inp)), unicodedata.normalize("NFC", unescape(want))
        if got != want:
            failures.append((rule, inp, got, want))
    assert not failures, "\n".join(map(repr, failures))


def test_lenient():
    # The nasal mark and same-word variants are forgiven; everything normalise() does still applies.
    assert lenient("जिसमें बहोत कि हैं।") == lenient("जिसमे बहुत की है") == "जिसमे बहुत की है"
    assert lenient("यह पे") == "ये पर"
    assert lenient("हिन्दी {Hindi}") == "हिदी"  # nasal + virama -> anusvara -> dropped
    assert lenient("घर") != lenient("घास")  # different words stay different
