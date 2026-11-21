# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["transformers", "torch", "torchaudio", "soundfile", "onnxruntime"]
# ///
"""Transcribe a manifest with AI4Bharat's IndicConformer (600M, multilingual).

    uv run runners/indicconformer.py MANIFEST OUT [--decoder ctc|rnnt]

Accept the model's terms on Hugging Face and set HF_TOKEN first. The model ships
its own loading code (trust_remote_code).
"""

import argparse
import json

import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel

from langs import INDICCONFORMER
from resume import finished

MODEL = "ai4bharat/indic-conformer-600m-multilingual"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--decoder", default="rnnt", choices=["ctc", "rnnt"])
    a = p.parse_args()

    done = finished(a.out)
    with open(a.manifest, encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r["id"] not in done]
    print(f"indicconformer: {len(done)} clips already done, {len(rows)} to go", flush=True)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True)
    with open(a.out, "a", encoding="utf-8") as out:
        for n, r in enumerate(rows, 1):
            hyp = None
            if lang := INDICCONFORMER.get(r["language"]):
                audio, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
                wav = torchaudio.functional.resample(torch.from_numpy(audio.mean(1))[None], sr, 16000)
                hyp = model(wav, lang, a.decoder)
            out.write(json.dumps({"id": r["id"], "model": f"indicconformer-{a.decoder}", "hyp": hyp}, ensure_ascii=False) + "\n")
            out.flush()
            if n % 100 == 0 or n == len(rows):
                print(f"indicconformer: {n}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
