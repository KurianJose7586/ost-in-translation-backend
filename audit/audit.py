"""M0 listening audit: are a district's errors in its speech or in its transcripts?

    uv run python audit/audit.py pick   # data/manifests/m0_audit.jsonl, 30 clips per district
    uv run python audit/audit.py page   # runs/audit/index.html, once the audio is in runs/audit/audio

Up to half of each district's clips are suspects: somewhere in the clip, all three
models wrote the same word in place of a reference word (or all dropped it), after
lenient normalisation. The rest are drawn at random from the district's other clips.
Each row keeps its stratum and the stratum's size, so a reference error rate found
in the sample can be weighted back to the whole district.

The page is one self-contained HTML file next to the audio folder: zip the folder,
send it to a listener, and they open index.html in a browser. No server needed.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from atlas_eval.normalize import lenient, normalise
from atlas_eval.score import align, read_jsonl, spaced

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/m0.jsonl"
AUDIT = ROOT / "data/manifests/m0_audit.jsonl"
RUNS = ROOT / "runs/m0"
OUT = ROOT / "runs/audit"
MODELS = {"whisper": "Whisper", "omnilingual": "Omnilingual", "indicconformer": "IndicConformer"}
PER_DISTRICT, SUSPECTS = 30, 15


def hyps():
    return {m: {r["id"]: r["hyp"] or "" for r in read_jsonl(RUNS / f"{m}.jsonl")} for m in MODELS}


def disputed(ref, hs):
    """Positions of reference words that every model replaced with the same word, or every model dropped."""
    seen = defaultdict(list)  # reference position -> what each model wrote there (None if dropped)
    for h in hs:
        pos = 0
        for op, _, w in align(ref, h):
            if op != "I":
                seen[pos].append(w if op != "C" else "=")
                pos += 1
    return {p for p, ws in seen.items() if len(ws) == len(hs) and "=" not in ws and len(set(ws)) == 1}


def pick():
    H = hyps()
    strata = defaultdict(lambda: ([], []))  # district -> (suspects, others)
    for c in read_jsonl(MANIFEST):
        ref = lenient(c["text"])
        strata[c["district"]][not disputed(ref, [lenient(H[m][c["id"]]) for m in MODELS])].append(c)
    rng, rows = random.Random(0), []
    for district, (suspects, others) in sorted(strata.items()):
        k = min(SUSPECTS, len(suspects))
        for name, pool, n in (("suspect", suspects, k), ("other", others, PER_DISTRICT - k)):
            rows += [{**c, "stratum": name, "stratum_size": len(pool)} for c in rng.sample(pool, n)]
        print(f"{district}: {len(suspects)} suspects, {len(others)} others; took {k} and {PER_DISTRICT - k}")
    with open(AUDIT, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{len(rows)} clips -> {AUDIT}")


def page():
    H = hyps()
    audio = {p.stem: p.name for p in (OUT / "audio").glob("*") if p.suffix != ".part"}
    rows = read_jsonl(AUDIT)
    missing = [r["id"] for r in rows if r["id"] not in audio]
    if missing:
        sys.exit(f"{len(missing)} clips have no audio in {OUT / 'audio'}, e.g. {missing[0]}")
    random.Random(1).shuffle(rows)  # listeners shouldn't be able to tell suspects from the rest
    rows.sort(key=lambda r: r["district"])  # stable: shuffled within each district
    clips = []
    for r in rows:
        ref = normalise(r["text"])
        loose = lenient(r["text"])
        flagged = disputed(loose, [lenient(H[m][r["id"]]) for m in MODELS])
        clips.append({
            "id": r["id"], "district": spaced(r["district"]), "state": spaced(r["state"]), "audio": f"audio/{audio[r['id']]}",
            "ref": ref, "disputed": sorted(flagged),  # lenient() keeps normalise()'s word positions
            "models": {name: align(ref, normalise(H[m][r["id"]])) for m, name in MODELS.items()},
        })
    data = json.dumps(clips, ensure_ascii=False).replace("</", "<\\/")
    html = (Path(__file__).parent / "listen.html").read_text(encoding="utf-8").replace("__CLIPS__", data)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"{len(clips)} clips -> {OUT / 'index.html'}")


if __name__ == "__main__":
    {"pick": pick, "page": page}[sys.argv[1]]()
