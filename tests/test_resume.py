import json
import subprocess
import sys
from pathlib import Path

RUNNERS = Path(__file__).parents[1] / "runners"


def todo(tmp_path, *flags):
    out = subprocess.run([sys.executable, str(RUNNERS / "resume.py"), str(tmp_path / "clips.jsonl"), str(tmp_path), *flags],
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return [json.loads(line)["id"] for line in out.splitlines()]


def test_only_clips_some_model_still_needs(tmp_path):
    clips = [("h1", "Hindi"), ("h2", "Hindi"), ("k1", "Kashmiri"), ("c1", "Chakma")]
    (tmp_path / "clips.jsonl").write_text("".join(json.dumps({"id": i, "language": lang}) + "\n" for i, lang in clips), encoding="utf-8")
    line = lambda i: json.dumps({"id": i, "model": "m", "hyp": None}) + "\n"
    for name in ("whisper", "omnilingual"):
        (tmp_path / f"{name}.jsonl").write_text(line("h1") + line("k1"), encoding="utf-8")
    (tmp_path / "indicconformer.jsonl").write_text(line("h1"), encoding="utf-8")

    # h2: nobody has it. k1: only IndicConformer supports Kashmiri, and it hasn't done it.
    assert todo(tmp_path) == ["h2", "k1"]
    # c1: no model supports Chakma, but every model still owes it a null line, so it runs without audio.
    assert todo(tmp_path, "--no-audio") == ["c1"]
