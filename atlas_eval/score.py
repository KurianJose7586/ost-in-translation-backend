"""Score transcripts per district and write the JSON the atlas site reads.

    atlas-eval MANIFEST TRANSCRIPTS... -o atlas.json [--round baseline]

District WER is total word errors over total reference words, not a mean of
clip WERs. Intervals come from a cluster bootstrap that resamples speakers,
because clips from one speaker aren't independent. `wer_pooled` shrinks each
district toward its language mean so a small district can't swing the map
colours; tables show the raw `wer`. `wer_lenient` scores the same clips with
`normalize.lenient`, which forgives spelling-only differences, using the same
speaker resamples for its interval.
"""

import argparse
import json
import re
from collections import Counter, defaultdict

import jiwer
import numpy as np

from atlas_eval.normalize import lenient, normalise

BOOTSTRAP = 1000
EXAMPLES = 3


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


def district_stats(clips, rng):
    """clips: (speaker, errors, words, lenient_errors, lenient_words, char_errors, chars) per clip."""
    per_speaker = defaultdict(lambda: np.zeros(4))
    for speaker, *c in clips:
        per_speaker[speaker] += c[:4]
    en = np.array(list(per_speaker.values()))
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


def spaced(s):
    """Vaani writes place names in CamelCase (WestChamparan); the site shows West Champaran."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)


def build(clips, hyps, round_name="baseline"):
    """clips: id -> manifest row. hyps: model -> id -> hypothesis (None = unsupported)."""
    rng = np.random.default_rng(0)
    refs = {i: normalise(c["text"]) for i, c in clips.items()}
    loose_refs = {i: lenient(c["text"]) for i, c in clips.items()}
    groups = defaultdict(list)
    for i, c in clips.items():
        groups[(c["state"], c["district"])].append(i)

    districts, by_language = [], defaultdict(list)
    for (state, name), ids in sorted(groups.items()):
        ids.sort()
        langs = Counter(clips[i]["language"] for i in ids)
        d = {
            "id": re.sub(r"[^a-z0-9]+", "-", f"{state}-{name}".lower()).strip("-"),
            "name": spaced(name),
            "state": spaced(state),
            "languages": {lang: n / len(ids) for lang, n in langs.most_common()},
            "results": {},
        }
        if polygons := clips[ids[0]].get("map"):
            d["map"] = polygons  # 2011 map polygons this district covers
        aligned = defaultdict(dict)
        for model, h in sorted(hyps.items()):
            seen = [i for i in ids if i in h]
            if not seen:
                continue
            scored = []
            for i in seen:
                if h[i] is None or not refs[i]:
                    continue
                hyp = normalise(h[i])
                rows = aligned[i][model] = align(refs[i], hyp)
                loose = counts(align(loose_refs[i], lenient(h[i])))
                scored.append((clips[i]["speaker"], *counts(rows), *loose, char_errors(refs[i], hyp), len(refs[i])))
            if not scored:
                d["results"][model] = {"status": "unsupported"}
                continue
            d["results"][model] = district_stats(scored, rng)
            by_language[(model, langs.most_common(1)[0][0])].append(d["results"][model])
        d["examples"] = [{"id": i, "ref": refs[i], "align": aligned[i]} for i in ids if aligned[i]][:EXAMPLES]
        districts.append(d)

    for stats in by_language.values():
        pool(stats)
    return {"version": 1, "round": round_name, "models": sorted(hyps), "districts": districts}


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
    a = p.parse_args(argv)

    clips = {c["id"]: c for c in read_jsonl(a.manifest)}
    hyps = defaultdict(dict)
    for path in a.transcripts:
        for t in read_jsonl(path):
            if t["id"] in clips:
                hyps[t["model"]][t["id"]] = t["hyp"]
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(_round(build(clips, hyps, a.round)), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
