"""M0 listening audit: are a district's errors in its speech or in its transcripts?

    uv run python audit/audit.py pick   # data/manifests/m0_audit.jsonl, 30 clips per district
    uv run python audit/audit.py page   # runs/audit/index.html, once the audio is in runs/audit/audio
    uv run python audit/audit.py score RESULTS.jsonl   # the file the listener sends back

Up to half of each district's clips are suspects: somewhere in the clip, all three
models wrote the same word in place of a reference word (or all dropped it), after
lenient normalisation. The rest are drawn at random from the district's other clips.
Each row keeps its stratum and the stratum's size, so a reference error rate found
in the sample can be weighted back to the whole district.

The page is one self-contained HTML file next to the audio folder: zip the folder,
send it to a listener, and they open index.html in a browser. No server needed.

`score` treats what the listener heard as the truth. Per district it prints the
reference WER (Vaani's transcript scored against the listener's) and the share
of clips with any reference error. Then, per model, the district's lenient WER
from runs/m0/atlas.json, and that WER corrected: minus the gap between scoring
the audited clips against Vaani and against the listener. Correcting the full
district's WER by the gap is steadier than rescoring 30 clips. Everything is
lenient, since listeners don't follow Vaani's spelling. Each audited clip is
weighted by its stratum's size over the clips audited from it, and intervals
redraw clips within each stratum. Clips marked "Can't tell" are left out.
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from atlas_eval.normalize import lenient, normalise
from atlas_eval.score import BOOTSTRAP, align, counts, read_jsonl, spaced

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


def estimate(strata, rng):
    """Weighted column totals for one district, and BOOTSTRAP resamples of them.

    strata: [(stratum_size, array of audited clips × columns)]. Each clip stands for
    stratum_size / audited clips of its stratum; resamples redraw clips within each stratum.
    """
    # ponytail: clips, not speakers, are resampled; fine while audited clips rarely share a speaker.
    point = sum(n / len(a) * a.sum(0) for n, a in strata)
    boot = sum(n / len(a) * a[rng.integers(0, len(a), (BOOTSTRAP, len(a)))].sum(1) for n, a in strata)
    return point, boot


def score(path):
    rows = {r["id"]: r for r in read_jsonl(AUDIT)}
    H = hyps()
    ids = {m: read_jsonl(RUNS / f"{m}.jsonl")[0]["model"] for m in MODELS}
    full = {d["name"]: d["results"] for d in json.loads((RUNS / "atlas.json").read_text(encoding="utf-8"))["districts"]}
    per, skipped = defaultdict(lambda: defaultdict(list)), Counter()
    for r in read_jsonl(path):
        m = rows.get(r["id"])
        if not m:
            continue
        heard, vaani = lenient(r["heard"]), lenient(m["text"])
        if r["unclear"] or not heard:
            skipped[m["district"]] += 1
            continue
        e, w = counts(align(heard, vaani))
        cols = [e, w, e > 0, 1, len(vaani.split())]  # reference errors, heard words, clip has one, clip, Vaani words
        for model in MODELS:
            hyp = lenient(H[model][r["id"]])
            cols += [counts(align(vaani, hyp))[0], counts(align(heard, hyp))[0]]
        per[(m["district"], m["state"])][m["stratum_size"], m["stratum"]].append(cols)

    rng, fixed = np.random.default_rng(0), defaultdict(dict)
    pct = lambda x: f"{x:.1%}"
    print("| District | Clips (can't tell) | Reference WER | Clips with an error | "
          + " | ".join(f"{n}: WER → corrected" for n in MODELS.values()) + " |")
    print("|---" * (4 + len(MODELS)) + "|")
    for (district, _), strata in sorted(per.items()):
        name = spaced(district)
        point, boot = estimate([(n, np.array(c, float)) for (n, _), c in strata.items()], rng)
        lo, hi = np.percentile(boot[:, 0] / boot[:, 1], [2.5, 97.5])
        cells = [f"{sum(map(len, strata.values()))} ({skipped[district]})",
                 f"{pct(point[0] / point[1])} ({pct(lo)}–{pct(hi)})", pct(point[2] / point[3])]
        for k, (m, label) in enumerate(MODELS.items()):
            ev, eh = 5 + 2 * k, 6 + 2 * k  # errors against Vaani, against the listener
            gap = point[ev] / point[4] - point[eh] / point[1]
            gaps = boot[:, ev] / boot[:, 4] - boot[:, eh] / boot[:, 1]
            r = full[name][ids[m]]
            # ponytail: the two errors are added as if independent; the audited clips are a subset of the full set.
            half = 1.96 * np.hypot(gaps.std(), (r["ci_lenient"][1] - r["ci_lenient"][0]) / 3.92)
            wer = r["wer_lenient"] - gap
            fixed[label][name] = wer, wer - half, wer + half
            cells.append(f"{pct(r['wer_lenient'])} → {pct(wer)} ({pct(wer - half)}–{pct(wer + half)})")
        print(f"| {name} | " + " | ".join(cells) + " |")

    print("\nGate on corrected WER:")
    for name, d in fixed.items():
        worst, best = max(d, key=lambda k: d[k][0]), min(d, key=lambda k: d[k][0])
        (w, lo, _), (b, _, hi) = d[worst], d[best]
        apart = "overlap" if lo <= hi else "don't overlap"
        print(f"  {name}: {worst} {pct(w)} ÷ {best} {pct(b)} = {w / b:.2f}×; intervals {apart}")


if __name__ == "__main__":
    {"pick": pick, "page": page, "score": score}[sys.argv[1]](*sys.argv[2:])
