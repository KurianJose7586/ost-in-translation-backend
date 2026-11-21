# /// script
# requires-python = ">=3.10"
# dependencies = ["faster-whisper"]
# ///
"""Transcribe a manifest with Whisper through faster-whisper.

    uv run runners/whisper.py MANIFEST OUT [--model large-v3] [--device auto]
"""

import argparse
import json

from faster_whisper import WhisperModel

from langs import WHISPER
from resume import finished


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    done = finished(a.out)
    model = WhisperModel(a.model, device=a.device, compute_type="default")
    with open(a.manifest, encoding="utf-8") as f, open(a.out, "a", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            if r["id"] in done:
                continue
            hyp = None
            if lang := WHISPER.get(r["language"]):
                segments, _ = model.transcribe(r["audio"], language=lang, beam_size=5)
                hyp = " ".join(s.text.strip() for s in segments)
            out.write(json.dumps({"id": r["id"], "model": f"whisper-{a.model}", "hyp": hyp}, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
