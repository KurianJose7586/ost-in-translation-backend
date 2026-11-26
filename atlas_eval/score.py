"""Score transcripts per district and language, and write the JSON the atlas site reads.

    atlas-eval MANIFEST TRANSCRIPTS... -o atlas.json [--round baseline] [--allow-partial]

A district is scored one language at a time: a Hindi WER and a Marathi WER averaged
together mean nothing, and a model may support only one of the two. District WER is
total word errors over total reference words, not a mean of clip WERs. Intervals come
from a cluster bootstrap that resamples speakers, because clips from one speaker aren't
independent; each district, language and model has its own seed, so adding a district
never moves another's interval. `wer_pooled` shrinks each district toward its
language's mean so a small district can't swing the map colours; tables show the raw
`wer`. `wer_lenient` scores the same clips with `normalize.lenient`, which forgives
spelling-only differences, using the same speaker resamples for its interval.

Every model must have a transcript for every clip, or models would be compared on
different clips; `--allow-partial` writes a progress round anyway, marked "partial".
Clips a runner couldn't read (an "error" field in its transcript) are left out for
every model. `reliability` is split-half: how well the district ranking from half of
each district's speakers predicts the ranking from the other half.
"""

import argparse
import json
import re
import sys
import zlib
from collections import Counter, defaultdict

import jiwer
import numpy as np

from atlas_eval.normalize import lenient, normalise

BOOTSTRAP = 1000
EXAMPLES = 3
SPLITS = 200  # random speaker splits for split-half reliability


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def align(ref, hyp):
    """Word alignment as [op, ref_word, hyp_word] rows; op is C, S, D or I."""
    if not hyp:
        return [["D", w, None] for w in ref.split()]
    out = jiwer.process_words(ref, hyp)
    r, h, rows = out.references[0], out.hypotheses[0], []
    for c in out.alignments[0]:
        rs, hs = r[c.ref_start_idx : c.ref_end_idx], h[c.hyp_start_idx : c.hyp_end_idx]
        if c.type == "delete":
            rows += [["D", w, None] for w in rs]
        elif c.type == "insert":
            rows += [["I", None, w] for w in hs]
        else:
            rows += [["C" if c.type == "equal" else "S", a, b] for a, b in zip(rs, hs)]
    return rows


def char_errors(ref, hyp):
    if not hyp:
        return len(ref)
    c = jiwer.process_characters(ref, hyp)
    return c.substitutions + c.deletions + c.insertions


def counts(rows):
    """(word errors, reference words) from an alignment."""
    return sum(op != "C" for op, _, _ in rows), sum(op != "I" for op, _, _ in rows)


def rank_corr(a, b):
    """Spearman's rank correlation (ties broken by order, which is fine for WERs)."""
    rank = lambda x: np.argsort(np.argsort(x))
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def speaker_counts(clips):
    """clips: (speaker, errors, words, lenient_errors, lenient_words, ...) -> speakers × 4 summed counts."""
    per_speaker = defaultdict(lambda: np.zeros(4))
    for speaker, *c in clips:
        per_speaker[speaker] += c[:4]
    return np.array(list(per_speaker.values()))


def district_stats(clips, rng):
    """clips: (speaker, errors, words, lenient_errors, lenient_words, char_errors, chars) per clip."""
    en = speaker_counts(clips)
    idx = rng.integers(0, len(en), (BOOTSTRAP, len(en)))
    boot = en[idx].sum(1)  # one row of summed [errors, words, lenient errors, lenient words] per resample
    strict, loose = boot[:, 0] / boot[:, 1], boot[:, 2] / boot[:, 3]
    total = en.sum(0)
    return {
        "wer": total[0] / total[1],
        "ci": list(np.percentile(strict, [2.5, 97.5])),
        "se": strict.std(),
        "wer_lenient": total[2] / total[3],
        "ci_lenient": list(np.percentile(loose, [2.5, 97.5])),
        "cer": sum(c[5] for c in clips) / sum(c[6] for c in clips),
        "clips": len(clips),
        "speakers": len(en),
    }


def pool(stats):
    """Shrink district WERs toward their mean (empirical Bayes, normal approximation)."""
    # ponytail: normal-normal shrinkage on bootstrap SEs; swap for the M1 mixed-effects model.
    wer = np.array([s["wer"] for s in stats])
    var = np.array([s.pop("se") ** 2 for s in stats])
    if len(stats) < 2:
        stats[0]["wer_pooled"] = stats[0]["wer"]
        return
    tau2 = max(wer.var(ddof=1) - var.mean(), 0.0)
    w = np.divide(tau2, tau2 + var, out=np.ones_like(var), where=tau2 + var > 0)
    for s, p in zip(stats, wer.mean() + w * (wer - wer.mean())):
        s["wer_pooled"] = p


def reliability(districts, rng, splits=SPLITS):
    """Split-half reliability of district WER, or None with fewer than 5 districts.

    districts: one speakers × 2+ array of [errors, words, ...] per district. Each split puts
    half of every district's speakers on each side; the two sides' district WERs are rank-
    correlated, averaged over splits, and Spearman-Brown corrected to the full sample size.
    """
    ds = [en for en in districts if len(en) >= 2]
    if len(ds) < 5:
        return None
    rs = []
    for _ in range(splits):
        a, b = [], []
        for en in ds:
            p = rng.permutation(len(en))
            x, y = en[p[: len(en) // 2]].sum(0), en[p[len(en) // 2 :]].sum(0)
            a.append(x[0] / x[1])
            b.append(y[0] / y[1])
        rs.append(rank_corr(a, b))
    r = float(np.mean(rs))
    return 2 * r / (1 + r)


def spaced(s):
    """Vaani writes place names in CamelCase (WestChamparan); the site shows West Champaran."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)


def seed(*key):
    return np.random.default_rng(zlib.crc32("/".join(key).encode()))


def examples(refs, aligned, clip_wer):
    """The EXAMPLES clips whose WER, averaged over models, is nearest the district's median clip."""
    mean = {i: float(np.mean(w)) for i, w in clip_wer.items()}
    if not mean:
        return []
    mid = float(np.median(list(mean.values())))
    pick = sorted(mean, key=lambda i: (abs(mean[i] - mid), i))[:EXAMPLES]
    return [{"id": i, "ref": refs[i], "align": aligned[i]} for i in sorted(pick)]


def build(clips, hyps, round_name="baseline"):
    """clips: id -> manifest row. hyps: model -> id -> hypothesis (None = unsupported)."""
    refs = {i: normalise(c["text"]) for i, c in clips.items()}
    loose_refs = {i: lenient(c["text"]) for i, c in clips.items()}
    groups = defaultdict(lambda: defaultdict(list))  # (state, district) -> language -> clip ids
    for i, c in clips.items():
        groups[c["state"], c["district"]][c["language"]].append(i)

    districts, by_language, halves = [], defaultdict(list), defaultdict(list)
    for (state, name), langs in sorted(groups.items()):
        did = re.sub(r"[^a-z0-9]+", "-", f"{state}-{name}".lower()).strip("-")
        total = sum(map(len, langs.values()))
        d = {
            "id": did,
            "name": spaced(name),
            "state": spaced(state),
            "languages": {lang: len(ids) / total for lang, ids in sorted(langs.items(), key=lambda kv: (-len(kv[1]), kv[0]))},
            "results": {},
            "examples": {},
        }
        if polygons := clips[next(iter(langs.values()))[0]].get("map"):
            d["map"] = polygons  # 2011 map polygons this district covers
        for lang, ids in sorted(langs.items()):
            ids.sort()
            results, aligned, clip_wer = {}, defaultdict(dict), defaultdict(list)
            for model, h in sorted(hyps.items()):
                scored = []
                for i in ids:
                    if i not in h or h[i] is None or not refs[i]:
                        continue
                    hyp = normalise(h[i])
                    rows = aligned[i][model] = align(refs[i], hyp)
                    e, n = counts(rows)
                    clip_wer[i].append(e / n)
                    loose = counts(align(loose_refs[i], lenient(h[i])))
                    scored.append((clips[i]["speaker"], e, n, *loose, char_errors(refs[i], hyp), len(refs[i])))
                if scored:
                    results[model] = district_stats(scored, seed(did, lang, model))
                    by_language[model, lang].append(results[model])
                    halves[model, lang].append(speaker_counts(scored))
                elif any(i in h for i in ids):
                    results[model] = {"status": "unsupported"}
            if results:
                d["results"][lang] = results
                d["examples"][lang] = examples(refs, aligned, clip_wer)
        districts.append(d)

    for stats in by_language.values():
        pool(stats)
    rel = defaultdict(dict)
    for (model, lang), ds in sorted(halves.items()):
        if (r := reliability(ds, seed("reliability", lang, model))) is not None:
            rel[lang][model] = r
    languages = Counter(c["language"] for c in clips.values())
    return {
        "version": 2,
        "round": round_name,
        "models": sorted(hyps),
        "languages": [lang for lang, _ in sorted(languages.items(), key=lambda kv: (-kv[1], kv[0])) if any(lang in d["results"] for d in districts)],
        "reliability": dict(rel),
        "districts": districts,
    }


def _round(x):
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def main(argv=None):
    p = argparse.ArgumentParser(prog="atlas-eval", description=__doc__.split("\n\n")[0])
    p.add_argument("manifest")
    p.add_argument("transcripts", nargs="+")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--round", default="baseline")
    p.add_argument("--allow-partial", action="store_true", help="write a round even if some model lacks some clips")
    a = p.parse_args(argv)

    clips = {c["id"]: c for c in read_jsonl(a.manifest)}
    hyps, failed = defaultdict(dict), set()
    for path in a.transcripts:
        for t in read_jsonl(path):
            if t["id"] in clips:
                hyps[t["model"]][t["id"]] = t["hyp"]
                if t.get("error"):
                    failed.add(t["id"])
    for model, h in sorted(hyps.items()):
        print(f"{model}: transcripts for {len(h)} of {len(clips)} clips", file=sys.stderr)
    partial = any(len(h) < len(clips) for h in hyps.values())
    if partial and not a.allow_partial:
        sys.exit("Some models lack transcripts for some clips, so they would be scored on different clips. "
                 "Finish the runs, or pass --allow-partial to write a progress round marked partial.")
    if failed:
        print(f"{len(failed)} clips a runner couldn't read are left out for every model", file=sys.stderr)
        clips = {i: c for i, c in clips.items() if i not in failed}
    out = build(clips, hyps, a.round)
    if partial:
        out["partial"] = True
    for lang, models in out["reliability"].items():
        print(f"split-half reliability, {lang}: " + ", ".join(f"{m} {r:.2f}" for m, r in models.items()), file=sys.stderr)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(_round(out), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
