# /// script
# requires-python = ">=3.10,<3.13"  # fairseq2n ships wheels only up to Python 3.12
# dependencies = ["omnilingual-asr"]
# ///
"""Transcribe a manifest with Meta's Omnilingual ASR. Linux only (fairseq2).

    uv run runners/omnilingual.py MANIFEST OUT [--model omniASR_LLM_3B_v2]

Clips must be shorter than 40 seconds. On a 16 GB Kaggle T4 the 3B models fit;
omniASR_LLM_7B_v2 needs about 17 GB, so run it on the DGX. Only the LLM models
take a language hint; CTC models get none.
"""

import argparse
import json

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from langs import OMNILINGUAL
from resume import finished

CHUNK = 64  # clips transcribed between saves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--model", default="omniASR_LLM_3B_v2")
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()

    done = finished(a.out)
    with open(a.manifest, encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r["id"] not in done]
    line = lambda r, hyp: json.dumps({"id": r["id"], "model": a.model, "hyp": hyp}, ensure_ascii=False) + "\n"
    todo = [r for r in rows if r["language"] in OMNILINGUAL]
    pipeline = ASRInferencePipeline(model_card=a.model) if todo else None
    with open(a.out, "a", encoding="utf-8") as out:
        out.writelines(line(r, None) for r in rows if r["language"] not in OMNILINGUAL)
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i : i + CHUNK]
            lang = [OMNILINGUAL[r["language"]] for r in chunk] if "LLM" in a.model else None
            hyps = pipeline.transcribe([r["audio"] for r in chunk], lang=lang, batch_size=a.batch)
            out.writelines(line(r, h) for r, h in zip(chunk, hyps))
            out.flush()


if __name__ == "__main__":
    main()
