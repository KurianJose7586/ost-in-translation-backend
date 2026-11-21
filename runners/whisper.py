# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "faster-whisper",
#   "av<19",  # PyAV 19 dropped an argument faster-whisper 1.2.1 passes
#   "nvidia-cublas-cu12; sys_platform == 'linux'",  # CTranslate2's GPU path needs CUDA 12 cuBLAS
#   "nvidia-cudnn-cu12==9.*; sys_platform == 'linux'",  # and cuDNN 9
# ]
# ///
"""Transcribe a manifest with Whisper through faster-whisper.

    uv run runners/whisper.py MANIFEST OUT [--model large-v3] [--device auto]
"""

import argparse
import json
import os
import sys


def _cuda_libs_on_path():
    """Point the loader at the pip copies of cuBLAS and cuDNN, restarting once so it sees them."""
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
    except ImportError:
        return  # not Linux: nothing to do
    dirs = [list(m.__path__)[0] for m in (nvidia.cublas.lib, nvidia.cudnn.lib)]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if not set(dirs) <= set(current.split(":")):
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + [current])
        os.execv(sys.executable, [sys.executable, *sys.argv])


_cuda_libs_on_path()

from faster_whisper import WhisperModel  # noqa: E402

from langs import WHISPER  # noqa: E402
from resume import finished  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    done = finished(a.out)
    with open(a.manifest, encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r["id"] not in done]
    print(f"whisper: {len(done)} clips already done, {len(rows)} to go", flush=True)
    model = WhisperModel(a.model, device=a.device, compute_type="default")
    with open(a.out, "a", encoding="utf-8") as out:
        for n, r in enumerate(rows, 1):
            hyp = None
            if lang := WHISPER.get(r["language"]):
                segments, _ = model.transcribe(r["audio"], language=lang, beam_size=5)
                hyp = " ".join(s.text.strip() for s in segments)
            out.write(json.dumps({"id": r["id"], "model": f"whisper-{a.model}", "hyp": hyp}, ensure_ascii=False) + "\n")
            out.flush()
            if n % 100 == 0 or n == len(rows):
                print(f"whisper: {n}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
