# /// script
# requires-python = ">=3.10"
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--model", default="omniASR_LLM_3B_v2")
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()

    with open(a.manifest, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    todo = [r for r in rows if r["language"] in OMNILINGUAL]
    lang = [OMNILINGUAL[r["language"]] for r in todo] if "LLM" in a.model else None
    pipeline = ASRInferencePipeline(model_card=a.model)
    hyps = dict(zip((r["id"] for r in todo), pipeline.transcribe([r["audio"] for r in todo], lang=lang, batch_size=a.batch)))
    with open(a.out, "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps({"id": r["id"], "model": a.model, "hyp": hyps.get(r["id"])}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
