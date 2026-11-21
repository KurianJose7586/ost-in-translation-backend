"""Vaani: index the files, cut a test slice, fetch the slice's audio.

    uv run --group data python data/vaani.py index Hindi runs/vaani/Hindi.json
    uv run --group data python data/vaani.py slice runs/vaani/Hindi.json data/m0_districts.tsv -o data/manifests/m0.jsonl
    uv run --group data python data/vaani.py fetch data/manifests/m0.jsonl runs/m0

Needs a Hugging Face token with access to ARTPARK-IISc/Vaani (accept its terms
first). Vaani is tens of terabytes and the Hub allows 5,000 requests per five
minutes, so nothing here downloads whole files: `index` reads one footer per
file, and `slice` and `fetch` read only the row groups and columns they need.
"""

import argparse
import bisect
import csv
import hashlib
import io
import json
import random
import struct
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate
from pathlib import Path

import pyarrow.parquet as pq

REPO = "ARTPARK-IISc/Vaani"
HERE = Path(__file__).parent
RATE = 10  # requests per second, well under the Hub's 5,000 per 5 minutes
MAX_SECONDS = 40  # Omnilingual ASR rejects longer clips
PER_SPEAKER = 8  # clips per test speaker, so each district gets many speakers
META = ["language", "duration", "speakerID", "gender", "state", "district", "isTranscriptionAvailable", "transcript",
        "UtteranceSequenceID", "referenceImage"]

_lock, _next = threading.Lock(), [0.0]


def _speaker(speaker_id):
    """Published manifests carry a hash: some Vaani speaker IDs embed fragments of names."""
    return hashlib.sha256(speaker_id.encode()).hexdigest()[:12]


def _get(path, start=0, end=None, tail=None):
    """One paced range request to a Vaani file; waits out rate limits."""
    from huggingface_hub import get_token

    headers = {"Authorization": f"Bearer {get_token()}", "Range": f"bytes=-{tail}" if tail else f"bytes={start}-{end - 1}"}
    while True:
        with _lock:
            now = time.monotonic()
            slot = _next[0] = max(_next[0], now) + 1 / RATE
        time.sleep(max(0.0, slot - now))
        try:
            req = urllib.request.Request(f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}", headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(int(e.headers.get("Retry-After") or 60))


def _footer(path):
    tail = _get(path, tail=1 << 17)
    (n,) = struct.unpack("<I", tail[-8:-4])
    if n + 8 > len(tail):
        tail = _get(path, tail=n + 8)
    return tail, pq.read_metadata(io.BytesIO(b"PAR1" + tail[-(n + 8):]))


class _Sparse(io.RawIOBase):
    """A file of known size where only some byte ranges were fetched; pyarrow reads only those."""

    def __init__(self, size, chunks):
        self.size, self.chunks, self.pos = size, chunks, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        for start, data in self.chunks:
            if start <= self.pos and self.pos + n <= start + len(data):
                b[:n] = data[self.pos - start : self.pos - start + n]
                self.pos += n
                return n
        raise OSError(f"bytes {self.pos}..{self.pos + n} were not fetched")


def _rows(path, size, columns, groups=None, rows=None):
    """Read `columns` for the given row groups (or the groups holding `rows`) with one request per group."""
    tail, md = _footer(path)
    starts = list(accumulate((md.row_group(g).num_rows for g in range(md.num_row_groups)), initial=0))
    if rows is not None:
        groups = sorted({bisect.bisect_right(starts, r) - 1 for r in rows})
    leaves = [i for i in range(md.num_columns) if md.schema.column(i).path.split(".")[0] in columns]
    chunks = [(size - len(tail), tail)]
    for g in groups:
        cols = [md.row_group(g).column(i) for i in leaves]
        offsets = [c.dictionary_page_offset if c.has_dictionary_page else c.data_page_offset for c in cols]
        lo, hi = min(offsets), max(o + c.total_compressed_size for o, c in zip(offsets, cols))
        chunks.append((lo, _get(path, lo, hi)))
    pf = pq.ParquetFile(_Sparse(size, chunks), pre_buffer=False)
    for g in groups:
        for i, r in enumerate(pf.read_row_group(g, columns=columns).to_pylist()):
            if rows is None or starts[g] + i in rows:
                yield {**r, "file": path, "row": starts[g] + i}


def _sizes(language):
    from huggingface_hub import HfApi

    return {f.path: f.size for f in HfApi().list_repo_tree(REPO, repo_type="dataset", path_in_repo=f"audio/{language}")}


def index(language, out):
    sizes = _sizes(language)

    def one(path):
        md = _footer(path)[1]
        d, t = md.schema.names.index("district"), md.schema.names.index("isTranscriptionAvailable")
        rgs = [md.row_group(g) for g in range(md.num_row_groups)]
        return path, [[rg.column(d).statistics.min, rg.column(d).statistics.max, rg.column(t).statistics.max == "Yes"] for rg in rgs]

    groups = {}
    with ThreadPoolExecutor(8) as ex:
        for n, (path, rgs) in enumerate(ex.map(one, sorted(sizes)), 1):
            groups[path] = rgs
            if n % 500 == 0:
                print(f"{n}/{len(sizes)} files")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"sizes": sizes, "row_groups": groups}, f)
    print(f"{len(groups)} files indexed -> {out}")


def cut(index_path, districts, out, per_district, seed):
    from atlas_eval.normalize import normalise

    with open(index_path, encoding="utf-8") as f:
        idx = json.load(f)
    with open(HERE / "vaani_districts.tsv", encoding="utf-8") as f:
        polygons = {(r["state"], r["district"]): r["map"].split(",") for r in csv.DictReader(f, delimiter="\t")}
    wanted = [tuple(line.split("\t")[:2]) for line in Path(districts).read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.startswith("#")]

    rng = random.Random(seed)
    manifest = []
    for state, district in wanted:
        groups = [(path, g) for path, rgs in sorted(idx["row_groups"].items())
                  for g, (lo, hi, transcribed) in enumerate(rgs) if lo == hi == district and transcribed]
        rng.shuffle(groups)
        by_speaker = defaultdict(list)
        # Read random row groups until there are about twice as many usable clips as needed.
        for b in range(0, len(groups), 16):
            by_file = defaultdict(list)
            for path, g in groups[b : b + 16]:
                by_file[path].append(g)
            with ThreadPoolExecutor(8) as ex:
                batches = ex.map(lambda kv: list(_rows(kv[0], idx["sizes"][kv[0]], META, groups=sorted(kv[1]))), by_file.items())
                for r in (r for batch in batches for r in batch):
                    if (r["state"] == state and r["isTranscriptionAvailable"] == "Yes" and r["duration"] < MAX_SECONDS
                            and r["speakerID"] not in (None, "", "NA") and normalise(r["transcript"] or "")):
                        by_speaker[r["speakerID"]].append(r)
            if sum(min(PER_SPEAKER, len(v)) for v in by_speaker.values()) >= 2 * per_district:
                break

        speakers = sorted(by_speaker)
        rng.shuffle(speakers)
        picked = []
        for s in speakers:
            if len(picked) >= per_district:
                break
            clips = sorted(by_speaker[s], key=lambda r: r["UtteranceSequenceID"])
            picked += rng.sample(clips, min(PER_SPEAKER, len(clips)))
        if len(picked) < per_district:
            print(f"warning: {district} has only {len(picked)} usable clips")
        for r in sorted(picked, key=lambda r: (r["speakerID"], r["UtteranceSequenceID"])):
            # UtteranceSequenceID counts utterances within one image description, so the image is part of the id.
            image = Path(r["referenceImage"] or "").stem.removeprefix("IISc_VaaniProject_")
            speaker = _speaker(r["speakerID"])
            manifest.append({
                "id": f"{speaker}-{image}-{r['UtteranceSequenceID']}", "audio": None, "text": r["transcript"],
                "language": r["language"], "state": state, "district": district, "speaker": speaker,
                "duration": round(r["duration"], 2), "image": r["referenceImage"],
                "source": {"file": r["file"], "row": r["row"]}, "map": polygons[(state, district)],
            })
        print(f"{district}: {len(picked)} clips from {len({r['speakerID'] for r in picked})} speakers")

    assert len({m["id"] for m in manifest}) == len(manifest), "clip ids are not unique"
    speakers_of = defaultdict(set)
    for m in manifest:
        speakers_of[m["speaker"]].add(m["district"])
    assert all(len(v) == 1 for v in speakers_of.values()), "a speaker appears in two districts"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest)
    print(f"{len(manifest)} clips, {len(speakers_of)} speakers, {sum(m['duration'] for m in manifest) / 3600:.1f} h -> {out}")


def fetch(manifest, out_dir):
    out = Path(out_dir).resolve()
    (out / "audio").mkdir(parents=True, exist_ok=True)
    with open(manifest, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    # Resume: clips already on disk from an earlier, interrupted run are kept, not fetched again.
    saved = {p.stem: p for p in (out / "audio").iterdir() if p.suffix != ".part"}
    by_file = defaultdict(dict)
    for r in rows:
        if r["id"] in saved:
            r["audio"] = str(saved[r["id"]])
        else:
            by_file[r["source"]["file"]][r["source"]["row"]] = r
    print(f"{len(rows) - sum(map(len, by_file.values()))} clips already saved, {sum(map(len, by_file.values()))} to fetch")
    sizes = {k: v for lang in {f.split("/")[1] for f in by_file} for k, v in _sizes(lang).items()}

    def one(path):
        for a in _rows(path, sizes[path], ["audio", "speakerID"], rows=set(by_file[path])):
            r = by_file[path][a["row"]]
            assert _speaker(a["speakerID"]) == r["speaker"], f"row mismatch for {r['id']}"
            dest = out / "audio" / (r["id"] + (Path(a["audio"]["path"] or "").suffix or ".wav"))
            part = dest.with_suffix(".part")  # a crash mid-write leaves a .part, never a half-written clip
            part.write_bytes(a["audio"]["bytes"])
            part.replace(dest)
            r["audio"] = str(dest)

    with ThreadPoolExecutor(8) as ex:
        for n, _ in enumerate(ex.map(one, sorted(by_file)), 1):
            if n % 20 == 0:
                print(f"{n}/{len(by_file)} files")
    with open(out / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{len(rows)} clips -> {out / 'manifest.jsonl'}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index")
    s.add_argument("language")
    s.add_argument("out")
    s = sub.add_parser("slice")
    s.add_argument("index")
    s.add_argument("districts")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--per-district", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("fetch")
    s.add_argument("manifest")
    s.add_argument("out_dir")
    a = p.parse_args()
    if a.cmd == "index":
        index(a.language, a.out)
    elif a.cmd == "slice":
        cut(a.index, a.districts, a.out, a.per_district, a.seed)
    else:
        fetch(a.manifest, a.out_dir)


if __name__ == "__main__":
    main()
