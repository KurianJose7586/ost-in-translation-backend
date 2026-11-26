# /// script
# requires-python = ">=3.10,<3.13"  # fairseq2n ships wheels only up to Python 3.12
# dependencies = [
#   "omnilingual-asr",
#   "torchaudio==2.8.*",  # fairseq2n pins torch 2.8; an unpinned torchaudio floats to a build that won't load
#   "pyarrow<26",  # fairseq2 pins NumPy 1.x, and pyarrow 26 refuses to load without NumPy 2
# ]
# [tool.uv]
# exclude-newer = "2026-11-24T00:00:00Z"  # the versions the benchmark ran with; a later rerun gets the same ones
# ///
"""Transcribe a manifest with Meta's Omnilingual ASR. Linux only (fairseq2).

    uv run runners/omnilingual.py MANIFEST OUT [--model omniASR_LLM_3B_v2] [--dtype bfloat16]

Clips must be shorter than 40 seconds. On a 16 GB Kaggle T4 the 3B models fit;
omniASR_LLM_7B_v2 needs about 17 GB, so run it on the DGX. Only the LLM models
take a language hint; CTC models get none.

M0 and the benchmark ran in bfloat16, the pipeline's default, which a T4 doesn't
support natively (7–12 s a clip). float16 may be much faster there, but it can change
the output, so its transcripts carry the model id `{model}-float16` and must match
bfloat16's WER on M0 clips before a round uses them (README, "Omnilingual in float16").
"""

import argparse
import json

import torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from langs import OMNILINGUAL
from resume import failed, finished

CHUNK = 64  # clips transcribed between saves


def transcribe(pipeline, chunk, model, batch):
    """Transcript lines for a chunk; if the chunk fails, each clip alone, so a bad file costs only itself."""
    lang = [OMNILINGUAL[r["language"]] for r in chunk] if "LLM" in model else None
    try:
        hyps = pipeline.transcribe([r["audio"] for r in chunk], lang=lang, batch_size=batch)
        return [{"id": r["id"], "model": model, "hyp": h} for r, h in zip(chunk, hyps)]
    except Exception as e:
        if len(chunk) == 1:
            return [{"id": chunk[0]["id"], "model": model} | failed(e)]
        return [line for r in chunk for line in transcribe(pipeline, [r], model, batch)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--model", default="omniASR_LLM_3B_v2")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    a = p.parse_args()
    model = a.model if a.dtype == "bfloat16" else f"{a.model}-{a.dtype}"

    done = finished(a.out)
    with open(a.manifest, encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r["id"] not in done]
    todo = [r for r in rows if r["language"] in OMNILINGUAL]
    print(f"omnilingual: {len(done)} clips already done, {len(todo)} to go", flush=True)
    pipeline = ASRInferencePipeline(model_card=a.model, dtype=getattr(torch, a.dtype)) if todo else None
    dump = lambda lines: out.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)
    with open(a.out, "a", encoding="utf-8") as out:
        dump({"id": r["id"], "model": model, "hyp": None} for r in rows if r["language"] not in OMNILINGUAL)
        for i in range(0, len(todo), CHUNK):
            dump(transcribe(pipeline, todo[i : i + CHUNK], model, a.batch))
            out.flush()
            print(f"omnilingual: {min(i + CHUNK, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
