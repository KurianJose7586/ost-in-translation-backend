"""M1 test sets from Vaani's official test split: survey it, check its images against train, draft manifests.

    uv run --group data python data/m1.py survey -o data/m1_test_survey.tsv
    uv run --group data python data/m1.py images Hindi Bengali
    uv run --group data python data/m1.py slice Hindi -o data/manifests/m1_hindi.jsonl

Reads ARTPARK-IISc/Vaani-transcription-part (accept its terms first). It has no speakerID,
duration or UtteranceSequenceID columns, so the speaker, image and timing come from the audio
file name, IISc_VaaniProject_{tag}_{state}_{district}_{speaker}_{session}_{image}_{start}_{end}.wav,
with times in milliseconds. The speakerID rebuilt from it hashes as in M0, so one person gets the
same hash in M0 and M1 manifests. Clip ids end in the start time, since there is no utterance
number. Raw rows hold unhashed speaker IDs and are cached in runs/vaani/, which is gitignored.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import vaani
from vaani import SPLITS, parse

ROOT = Path(__file__).parents[1]
MIN_CLIPS = 100
# M0 took up to 8 clips per speaker, but the test split holds every clip of a speaker, so 8 would leave
# a median of 27 speakers per Hindi district; 4 gives 42 and keeps 93 of 99 districts (notes/decisions.md).
PER_SPEAKER = 4


def _scan(language, split, columns):
    """Every row of one language's files in one split, reading only `columns`."""
    sizes = {p: s for p, s in vaani._sizes(SPLITS, language).items() if p.rsplit("/", 1)[1].startswith(split + "-")}

    def one(path):
        groups = range(vaani._footer(SPLITS, path)[1].num_row_groups)
        return list(vaani._rows(SPLITS, path, sizes[path], columns, groups=groups))

    with ThreadPoolExecutor(8) as ex:
        return [r for batch in ex.map(one, sorted(sizes)) for r in batch]


def _cached(path, build):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def test_rows(language):
    return _cached(ROOT / f"runs/vaani/{language}_test.json", lambda: [
        {"name": r["audio"]["path"], "file": r["file"], "row": r["row"], "language": r["language"], "state": r["state"],
         "district": r["district"], "transcript": r["transcript"], "image": r["referenceImage"]}
        for r in _scan(language, "test", ["audio.path", "language", "state", "district", "transcript", "referenceImage"])])


def train_images(language):
    return set(_cached(ROOT / f"runs/vaani/{language}_train_images.json",
                       lambda: sorted({r["referenceImage"] for r in _scan(language, "train", ["referenceImage"])} - {None})))


def train_image_counts(language):
    """How many train clips describe each image."""
    return Counter(_cached(ROOT / f"runs/vaani/{language}_train_image_counts.json",
                           lambda: Counter(r["referenceImage"] for r in _scan(language, "train", ["referenceImage"]) if r["referenceImage"])))


def usable(clips):
    """M0's rules: {(state, district): {speaker: [clip, ...]}} for clips with a parsed speaker, under
    40 s and a non-empty normalised transcript, from speakers who appear in only one district."""
    from atlas_eval.normalize import normalise

    by = defaultdict(lambda: defaultdict(list))
    for c in clips:
        p = parse(c["name"])
        if p and p[3] < vaani.MAX_SECONDS and normalise(c["transcript"] or ""):
            by[c["state"], c["district"]][p[0]].append(c)
    seen = Counter(s for speakers in by.values() for s in speakers)
    # An ID in two districts is two people whose districts share their first letters (BIPA_d in Patna
    # and Paschim), or one person in two districts; either way M0's rule leaves them out. Say how many.
    if dropped := sum(n > 1 for n in seen.values()):
        print(f"{dropped} speaker IDs appear in more than one district and are left out")
    return {d: {s: v for s, v in speakers.items() if seen[s] == 1} for d, speakers in by.items()}


def capped(speakers):
    return sum(min(PER_SPEAKER, len(v)) for v in speakers.values())


def _polygons():
    with open(vaani.HERE / "vaani_districts.tsv", encoding="utf-8") as f:
        return {(r["state"], r["district"]): r["map"].split(",") for r in csv.DictReader(f, delimiter="\t")}


def _test_languages():
    from huggingface_hub import HfApi

    files = HfApi().list_repo_tree(SPLITS, repo_type="dataset", path_in_repo="audio", recursive=True)
    return sorted({f.path.split("/")[1] for f in files if f.path.rsplit("/", 1)[-1].startswith("test-")})


def survey(out):
    polygons, lines = _polygons(), []
    print("| Language | Districts | ≥100 usable | ≥150 usable | Usable clips | Hours |\n|---|---|---|---|---|---|")
    for language in _test_languages():
        clips = test_rows(language)
        ok = usable(clips)
        by = defaultdict(list)
        for c in clips:
            by[c["state"], c["district"]].append(c)
        n = Counter()
        for (state, district), cs in sorted(by.items()):
            parsed = [p for p in map(parse, (c["name"] for c in cs)) if p]
            u = ok.get((state, district), {})
            lines.append([language, state, district, len(cs), len({p[0] for p in parsed}),
                          round(sum(p[3] for p in parsed) / 3600, 2), capped(u), len(u),
                          sum(c["language"] != language for c in cs), (state, district) in polygons])
            n["usable"] += capped(u)
            n["hours"] += sum(p[3] for p in parsed) / 3600
            n[100] += capped(u) >= MIN_CLIPS
            n[150] += capped(u) >= 150
        print(f"| {language} | {len(by)} | {n[100]} | {n[150]} | {n['usable']:,} | {n['hours']:.1f} |")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["language", "state", "district", "clips", "speakers", "hours", "usable_clips", "usable_speakers",
                    "other_language_rows", "on_map"])
        w.writerows(lines)
    print(f"{len(lines)} language-districts -> {out}")


def images(languages):
    for language in languages:
        train, clips = train_images(language), test_rows(language)
        for kind in ("GENERIC", "SPECIFIC"):
            mine = [c["image"] for c in clips if c["image"] and kind in c["image"]]
            shared = {i for i in mine if i in train}
            print(f"{language} {kind}: {len(shared)} of {len(set(mine))} test images are also in train "
                  f"({sum(i in train for i in mine) / max(1, len(mine)):.0%} of {len(mine)} clips)")
        # The other side: how much training data keeping the M1 draft's images out of training would cost.
        draft = ROOT / f"data/manifests/m1_{language.lower()}.jsonl"
        if draft.exists():
            used = {json.loads(line)["image"] for line in open(draft, encoding="utf-8")} - {None}
            counts = train_image_counts(language)
            n = sum(counts.values())
            for kind in ("GENERIC", "SPECIFIC"):
                k = sum(v for i, v in counts.items() if i in used and kind in i)
                print(f"{language} {kind}: {k:,} of {n:,} train clips ({k / n:.1%}) describe an image the M1 draft uses")


def cut(language, out, per_district, seed):
    polygons, manifest, train = _polygons(), [], train_images(language)
    for (state, district), speakers in sorted(usable(test_rows(language)).items()):
        if capped(speakers) < MIN_CLIPS:
            continue
        rng = random.Random(f"{seed}-{state}-{district}")  # adding a district leaves the others' samples alone
        order = sorted(speakers)
        rng.shuffle(order)
        picked = []
        for s in order:
            if len(picked) >= per_district:
                break
            clips = sorted(speakers[s], key=lambda c: c["name"])
            picked += [(s, c) for c in rng.sample(clips, min(PER_SPEAKER, len(clips)))]
        if (state, district) not in polygons:
            print(f"warning: {state}/{district} is not in vaani_districts.tsv; it has no map polygon")
        for s, c in sorted(picked, key=lambda sc: sc[1]["name"]):
            _, image, start, seconds = parse(c["name"])
            speaker = vaani._speaker(s)
            manifest.append({
                "id": f"{speaker}-{image}-{start}", "audio": None, "text": c["transcript"], "language": c["language"],
                "state": state, "district": district, "speaker": speaker, "duration": round(seconds, 2),
                "image": c["image"], "image_in_train": c["image"] in train,  # Kaan is scored on both sides (decisions.md)
                "source": {"repo": SPLITS, "file": c["file"], "row": c["row"]},
                "map": polygons.get((state, district), []),
            })
        print(f"{district}: {len(picked)} clips from {len({s for s, _ in picked})} speakers")

    assert len({m["id"] for m in manifest}) == len(manifest), "clip ids are not unique"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest)
    print(f"{len(manifest)} clips, {len({m['speaker'] for m in manifest})} speakers, "
          f"{len({m['district'] for m in manifest})} districts, {sum(m['duration'] for m in manifest) / 3600:.1f} h -> {out}")
    print(f"SHA-256 {hashlib.sha256(Path(out).read_bytes()).hexdigest()}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey")
    s.add_argument("-o", "--out", required=True)
    s = sub.add_parser("images")
    s.add_argument("languages", nargs="+")
    s = sub.add_parser("slice")
    s.add_argument("language")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--per-district", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.cmd == "survey":
        survey(a.out)
    elif a.cmd == "images":
        images(a.languages)
    else:
        cut(a.language, a.out, a.per_district, a.seed)


if __name__ == "__main__":
    main()
