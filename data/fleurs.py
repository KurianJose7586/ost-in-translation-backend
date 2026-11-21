"""Build a smoke-test manifest from FLEURS Hindi (CC BY 4.0, no gate).

    python data/fleurs.py OUT_DIR [--n 50] [--split dev]

Streams the split's audio archive and stops after N clips, so it downloads
only what it needs. FLEURS has no speaker IDs, so each clip counts as its own
speaker, and every clip goes in one pseudo-district called FLEURS.
"""

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/hi_in/"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--split", default="dev")
    a = p.parse_args()

    out = Path(a.out_dir).resolve()
    (out / "audio").mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE + f"{a.split}.tsv") as r:
        text = {f[1]: f[2] for f in (line.split("\t") for line in r.read().decode("utf-8").splitlines())}

    rows = []
    with urllib.request.urlopen(BASE + f"audio/{a.split}.tar.gz") as r, tarfile.open(fileobj=r, mode="r|gz") as tar:
        for m in tar:
            name = Path(m.name).name
            if not m.isfile() or name not in text:
                continue
            path = out / "audio" / name
            path.write_bytes(tar.extractfile(m).read())
            rows.append({"id": path.stem, "audio": str(path), "text": text[name], "language": "Hindi",
                         "district": "FLEURS", "state": "FLEURS", "speaker": path.stem})
            if len(rows) == a.n:
                break

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{len(rows)} clips -> {out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
