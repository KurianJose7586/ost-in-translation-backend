"""Vaani-Benchmark-V1.0: fetch one language as a manifest, and score the baselines on it.

    uv run --group data python data/benchmark.py fetch runs/bench [--language Hindi]
    uv run python data/benchmark.py score runs/bench/manifest.jsonl runs/bench/*.jsonl [--m0 runs/m0]

The benchmark (ARTPARK-IISc/Vaani-Benchmark-V1.0; accept its terms first) is held
out from both Vaani repositories and was published after IndicConformer, so it
can show training on any Vaani release. Every segment has three independent
transcriptions.

`fetch` saves the audio and writes a manifest the runners read: `text` is
transcription_v1, `refs` holds all three, speaker IDs are hashed, and segments
of 40 s or more are dropped, as in M0. Don't commit it: the benchmark is gated.

`score` prints, for each model:
- the benchmark's own WER (its text cleaning and three-reference alignment), to
  check our runs against the published leaderboard;
- WER and lenient WER against transcription_v1 with our normaliser, as in M0;
- per district, how much the transcribers disagree (transcriptions 2 and 3
  scored against 1), next to each model's WER;
- with --m0 (a folder of M0 transcripts), each model's benchmark WER over its
  M0 WER. A model trained on Vaani does better on M0 than on the benchmark,
  relative to models that weren't; the same clips go to every model, so a ratio
  of those ratios cancels how hard each set is. Intervals resample speakers.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from jiwer import process_words

from atlas_eval.normalize import lenient, normalise
from atlas_eval.score import BOOTSTRAP, align, counts, read_jsonl, spaced

REPO = "ARTPARK-IISc/Vaani-Benchmark-V1.0"
M0 = Path(__file__).parent / "manifests/m0.jsonl"
TRANSCRIBERS = ["transcriber 2", "transcriber 3"]


def fetch(out_dir, language):
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    from vaani import MAX_SECONDS, _speaker

    out = Path(out_dir).resolve()
    (out / "audio").mkdir(parents=True, exist_ok=True)
    folder = f"data/{language.lower()}"
    root = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=f"{folder}/*"))
    rows, n, long = [], 0, 0
    for path in sorted(root.glob(f"{folder}/*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for r in batch.to_pylist():
                n += 1
                if r["duration"] >= MAX_SECONDS:
                    long += 1
                    continue
                speaker = _speaker(str(r["speakerID"]))
                dest = out / "audio" / f"{speaker}-{n:05d}.wav"
                if not dest.exists():
                    part = dest.with_suffix(".part")  # a crash mid-write never leaves a half-written clip
                    part.write_bytes(r["audio"]["bytes"])
                    part.replace(dest)
                refs = [r[f"transcription_v{k}"] or "" for k in (1, 2, 3)]
                rows.append({"id": dest.stem, "audio": str(dest), "text": refs[0], "refs": refs, "language": r["language"],
                             "state": r["state"], "district": r["district"], "speaker": speaker,
                             "duration": r["duration"], "image": r["reference_image"]})
    with open(out / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{n} segments, {long} of {MAX_SECONDS} s or more dropped, {len(rows)} -> {out / 'manifest.jsonl'}")


# clean_text and approach3_wer: the benchmark's own scoring, from run_eval.py in
# github.com/ARTPARK-Speech-Models/Vaani-Benchmark-Evaluation-Toolkit
# (MIT License, Copyright (c) 2026 ARTPARK-Speech-Models), condensed.
def clean_text(text):
    text = re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}|<.*?>|❴.*?❵", "", text or "")
    text = re.sub(r"[,?.!;:\"%'‘’“”—–\-।]", "", text)
    return re.sub(r"\s+", " ", text).strip() or "NA"


def approach3_wer(refs1, refs2, refs3, hyps):
    """A hypothesis word is an error only if no reference marks it correct; a deletion only if all three delete it."""
    subs = ins = dels = length = 0
    for r1, r2, r3, hyp in zip(refs1, refs2, refs3, hyps):
        labels, deleted = defaultdict(list), []
        for ref in (r1, r2, r3):
            out = process_words(ref, hyp)
            hyp_words, ref_words, gone = out.hypotheses[0], out.references[0], set()
            for c in out.alignments[0]:
                if c.type == "delete":
                    gone.update(ref_words[c.ref_start_idx : c.ref_end_idx])
                else:
                    kind = {"equal": "correct", "insert": "insert", "substitute": "substitute"}[c.type]
                    for i in range(c.hyp_start_idx, c.hyp_end_idx):
                        labels[i].append(kind)
            deleted.append(gone)
        d = len(set.intersection(*deleted))
        s = sum("correct" not in v and "substitute" in v for v in labels.values())
        n = sum("correct" not in v and "substitute" not in v and "insert" in v for v in labels.values())
        subs, ins, dels, length = subs + s, ins + n, dels + d, length + d - n + len(hyp_words)
    return (subs + ins + dels) / length


def transcripts(paths):
    hyps = defaultdict(dict)
    for path in paths:
        if Path(path).name != "manifest.jsonl":
            for t in read_jsonl(path):
                hyps[t["model"]][t["id"]] = t["hyp"] or ""
    return hyps


def clip_counts(clips, hyps, models):
    """clips × models × [errors, words, lenient errors, lenient words], each against the clip's `text`."""
    return np.array([[[*counts(align(normalise(c["text"]), normalise(hyps[m][c["id"]]))),
                       *counts(align(lenient(c["text"]), lenient(hyps[m][c["id"]])))] for m in models] for c in clips])


def group(clips, a, key):
    ix = defaultdict(list)
    for i, c in enumerate(clips):
        ix[key(c)].append(i)
    return {k: a[v].sum(0) for k, v in ix.items()}


def speaker_wer(clips, a, rng):
    """WER and lenient WER per model (models × 2), and the same over speaker resamples (BOOTSTRAP × models × 2)."""
    s = np.array(list(group(clips, a, lambda c: c["speaker"]).values()))
    b = s[rng.integers(0, len(s), (BOOTSTRAP, len(s)))].sum(1)
    t = s.sum(0)
    return t[:, [0, 2]] / t[:, [1, 3]], b[..., [0, 2]] / b[..., [1, 3]]


def scorable(clips, hyps, models):
    return [c for c in clips if normalise(c["text"]) and lenient(c["text"]) and all(c["id"] in hyps[m] for m in models)]


def score(manifest, paths, m0_dir):
    hyps, rng = transcripts(paths), np.random.default_rng(0)
    models = sorted(hyps)
    clips = read_jsonl(manifest)
    for c in clips:
        for name, ref in zip(TRANSCRIBERS, c["refs"][1:]):
            hyps[name][c["id"]] = ref
    clips = scorable(clips, hyps, models)
    print(f"{len(clips)} segments from {len({c['speaker'] for c in clips})} speakers scored\n")

    a = clip_counts(clips, hyps, models + TRANSCRIBERS)
    point, boot = speaker_wer(clips, a, rng)
    refs = [[clean_text(c["refs"][k]) for c in clips] for k in range(3)]
    print("| Model | Benchmark WER (3 references) | WER vs transcription 1 | Lenient |\n|---|---|---|---|")
    for k, m in enumerate(models + TRANSCRIBERS):
        bench = approach3_wer(*refs, [clean_text(hyps[m][c["id"]]) for c in clips]) if m in models else None
        lo, hi = np.percentile(boot[:, k, 0], [2.5, 97.5])
        print(f"| {m} | {'' if bench is None else f'{bench:.1%}'} | {point[k, 0]:.1%} ({lo:.1%}–{hi:.1%}) | {point[k, 1]:.1%} |")

    districts = group(clips, a, lambda c: (c["state"], c["district"]))
    sizes = group(clips, np.ones(len(clips)), lambda c: (c["state"], c["district"]))
    speakers = defaultdict(set)
    for c in clips:
        speakers[c["state"], c["district"]].add(c["speaker"])
    wer = {d: v[:, 0] / v[:, 1] for d, v in districts.items()}  # per model, then transcribers 2 and 3
    disagree = {d: v[len(models):, 0].sum() / v[len(models):, 1].sum() for d, v in districts.items()}
    order = sorted(wer, key=lambda d: -wer[d][: len(models)].min())
    print(f"\n| District | Clips | Speakers | Transcribers disagree | {' | '.join(models)} |\n|---|---|---|---|{'---|' * len(models)}")
    for d in order:
        cells = " | ".join(f"{x:.1%}" for x in wer[d][: len(models)])
        print(f"| {spaced(d[1])}, {spaced(d[0])} | {sizes[d]:.0f} | {len(speakers[d])} | {disagree[d]:.1%} | {cells} |")
    rank = lambda x: np.argsort(np.argsort(x))
    for k, m in enumerate(models):
        rho = np.corrcoef(rank([disagree[d] for d in order]), rank([wer[d][k] for d in order]))[0, 1]
        print(f"Rank correlation across districts, transcriber disagreement vs {m}: {rho:.2f}")

    if m0_dir:
        old = transcripts(sorted(Path(m0_dir).glob("*.jsonl")))
        shared = [m for m in models if m in old]
        m0 = scorable(read_jsonl(M0), old, shared)
        m0_point, m0_boot = speaker_wer(m0, clip_counts(m0, old, shared), rng)
        ix = [models.index(m) for m in shared]
        ratio, rboot = point[ix] / m0_point, boot[:, ix] / m0_boot  # above 1 = worse on the benchmark
        print(f"\n{len(m0)} M0 clips. Benchmark WER (vs transcription 1) over M0 WER:\n")
        print("| Model | M0 WER | Benchmark WER | Benchmark ÷ M0 | Lenient, benchmark ÷ M0 |\n|---|---|---|---|---|")
        for k, m in enumerate(shared):
            ci = [np.percentile(rboot[:, k, j], [2.5, 97.5]) for j in (0, 1)]
            print(f"| {m} | {m0_point[k, 0]:.1%} | {point[ix[k], 0]:.1%} | {ratio[k, 0]:.2f} ({ci[0][0]:.2f}–{ci[0][1]:.2f}) "
                  f"| {ratio[k, 1]:.2f} ({ci[1][0]:.2f}–{ci[1][1]:.2f}) |")
        for i in range(len(shared)):
            for j in range(i + 1, len(shared)):
                for e, label in ((0, "WER"), (1, "lenient")):
                    lo, hi = np.percentile(rboot[:, i, e] / rboot[:, j, e], [2.5, 97.5])
                    print(f"  {shared[i]} ÷ {shared[j]}, {label}: {ratio[i, e] / ratio[j, e]:.2f} ({lo:.2f}–{hi:.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("out_dir")
    f.add_argument("--language", default="Hindi")
    s = sub.add_parser("score")
    s.add_argument("manifest")
    s.add_argument("transcripts", nargs="+")
    s.add_argument("--m0", help="folder of M0 transcripts, scored against data/manifests/m0.jsonl")
    a = p.parse_args()
    if a.cmd == "fetch":
        fetch(a.out_dir, a.language)
    else:
        score(a.manifest, a.transcripts, a.m0)


if __name__ == "__main__":
    main()
