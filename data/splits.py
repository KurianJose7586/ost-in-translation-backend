"""Which of Vaani's official splits each clip of a manifest belongs to, and a contamination check.

    uv run --group data python data/splits.py match data/manifests/m0.jsonl -o data/manifests/m0_splits.jsonl
    uv run python data/splits.py compare data/manifests/m0.jsonl data/manifests/m0_splits.jsonl runs/m0/*.jsonl

`compare` scores each model on clips from the official train split and on held-out
clips (validation and test). A model that trained on Vaani's train split does
better on train clips than on held-out ones, relative to models that didn't; the
same clips go to every model, so a ratio of ratios cancels how hard the clips are.
Intervals resample speakers within each side.

Vaani's official train, validation and test split is in ARTPARK-IISc/Vaani-transcription-part
(accept its terms first), not in ARTPARK-IISc/Vaani, where M0 was drawn. Clips are
matched by audio file name, IISc_VaaniProject_{tag}_{state}_{district}_{speaker}_{image}_{start}_{end}.wav,
which both repositories share; the manifest's source row gives each clip's name.
A clip missing from the split repository takes its speaker's split, since the
split keeps each speaker in one split. Scanning the split repository reads the
small columns of most of its row groups (about 10 minutes for Hindi), so the rows
for the manifest's districts are cached in runs/vaani/.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import vaani

MAIN, SPLITS = "ARTPARK-IISc/Vaani", "ARTPARK-IISc/Vaani-transcription-part"
NAME = re.compile(r"IISc_VaaniProject_[A-Z]+_([A-Za-z]+)_([A-Za-z]+)_(.+?)_((?:[A-Za-z]+-SPECIFIC|GENERIC)_\d+)_\d+_\d+\.wav")


def speaker(name):
    return NAME.fullmatch(name).groups()[:3]  # state, district, speaker as written in file names


def names(clips, language):
    """Audio file name of each manifest clip, read from its source row in ARTPARK-IISc/Vaani."""
    vaani.REPO = MAIN  # vaani's readers use this module-level repository
    sizes, rows = vaani._sizes(language), defaultdict(set)
    for c in clips:
        rows[c["source"]["file"]].add(c["source"]["row"])
    with ThreadPoolExecutor(8) as ex:
        found = ex.map(lambda kv: list(vaani._rows(kv[0], sizes[kv[0]], ["audio.path"], rows=kv[1])), rows.items())
        got = {(r["file"], r["row"]): r["audio"]["path"] for batch in found for r in batch}
    return [got[c["source"]["file"], c["source"]["row"]] for c in clips]


def official(language, districts, cache):
    """Rows of the split repository in the given districts: {name, split, transcript}."""
    if cache.exists():
        return [json.loads(line) for line in open(cache, encoding="utf-8")]
    vaani.REPO = SPLITS
    sizes = vaani._sizes(language)

    def one(path):
        md = vaani._footer(path)[1]
        d = md.schema.names.index("district")
        stats = [md.row_group(g).column(d).statistics for g in range(md.num_row_groups)]
        groups = [g for g, s in enumerate(stats) if any(s.min <= x <= s.max for x in districts)]
        split = path.rsplit("/", 1)[1].split("-")[0]
        return [{"name": r["audio"]["path"], "split": split, "transcript": r["transcript"]}
                for r in vaani._rows(path, sizes[path], ["audio.path", "district", "transcript"], groups=groups)
                if r["district"] in districts]

    with ThreadPoolExecutor(8) as ex:
        rows = [r for batch in ex.map(one, sorted(sizes)) for r in batch]
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return rows


def match(manifest, out, language):
    clips = [json.loads(line) for line in open(manifest, encoding="utf-8")]
    rows = official(language, {c["district"] for c in clips},
                    Path(__file__).parents[1] / f"runs/vaani/{language}_splits_{Path(manifest).stem}.jsonl")
    by_name = {r["name"]: r for r in rows}
    by_speaker = defaultdict(set)
    for r in rows:
        by_speaker[speaker(r["name"])].add(r["split"])

    found, tally = [], Counter()
    for c, name in zip(clips, names(clips, language)):
        if r := by_name.get(name):
            split, how = r["split"], "same transcript" if r["transcript"] == c["text"] else "new transcript"
        else:
            s = by_speaker.get(speaker(name), set())
            split, how = (next(iter(s)) if len(s) == 1 else "several" if s else "none"), "speaker"
        found.append({"id": c["id"], "split": split, "matched": how})
        tally[split, how] += 1
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r) + "\n" for r in found)
    for (split, how), n in sorted(tally.items()):
        print(f"{split:>10}, matched by {how}: {n}")
    print(f"{len(found)} clips -> {out}")


def compare(manifest, splits, transcripts):
    import numpy as np

    from atlas_eval.normalize import lenient, normalise
    from atlas_eval.score import BOOTSTRAP, align, counts, read_jsonl

    train = {r["id"]: r["split"] == "train" for r in read_jsonl(splits) if r["split"] in ("train", "validation", "test")}
    hyps = defaultdict(dict)
    for path in transcripts:
        for t in read_jsonl(path):
            hyps[t["model"]][t["id"]] = t["hyp"] or ""
    models = sorted(hyps)
    per = defaultdict(lambda: np.zeros((len(models), 4)))  # (train?, speaker) -> per model: errors, words, lenient errors, lenient words
    clips = Counter()
    for c in read_jsonl(manifest):
        if c["id"] not in train:
            continue
        clips[train[c["id"]]] += 1
        for k, m in enumerate(models):
            h = hyps[m][c["id"]]
            per[train[c["id"]], c["speaker"]][k] += [*counts(align(normalise(c["text"]), normalise(h))),
                                                     *counts(align(lenient(c["text"]), lenient(h)))]
    rng = np.random.default_rng(0)
    sides = {}
    for side in (True, False):
        a = np.array([v for (t, _), v in per.items() if t == side])
        sides[side] = a.sum(0), a[rng.integers(0, len(a), (BOOTSTRAP, len(a)))].sum(1)  # point, resamples
    print(f"{clips[True]} train clips, {clips[False]} held-out clips")
    for e, label in ((0, "WER"), (2, "Lenient WER")):
        wer = {s: (p[:, e] / p[:, e + 1], b[..., e] / b[..., e + 1]) for s, (p, b) in sides.items()}
        ratio, boot = wer[True][0] / wer[False][0], wer[True][1] / wer[False][1]  # per model; below 1 = better on train
        print(f"\n| {label} | Train | Held out | Train ÷ held out |\n|---|---|---|---|")
        for k, m in enumerate(models):
            lo, hi = np.percentile(boot[:, k], [2.5, 97.5])
            print(f"| {m} | {wer[True][0][k]:.1%} | {wer[False][0][k]:.1%} | {ratio[k]:.2f} ({lo:.2f}–{hi:.2f}) |")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                lo, hi = np.percentile(boot[:, i] / boot[:, j], [2.5, 97.5])
                print(f"  {models[i]} ÷ {models[j]}: {ratio[i] / ratio[j]:.2f} ({lo:.2f}–{hi:.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match")
    m.add_argument("manifest")
    m.add_argument("-o", "--out", required=True)
    m.add_argument("--language", default="Hindi")
    c = sub.add_parser("compare")
    c.add_argument("manifest")
    c.add_argument("splits")
    c.add_argument("transcripts", nargs="+")
    a = p.parse_args()
    if a.cmd == "match":
        match(a.manifest, a.out, a.language)
    else:
        compare(a.manifest, a.splits, a.transcripts)


if __name__ == "__main__":
    main()
