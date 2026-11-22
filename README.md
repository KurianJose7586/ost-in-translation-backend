# Lost in Transcription: backend

Evaluation, data and (later) training code for [Lost in Transcription](https://github.com/KurianJose7586/lost-in-translation-frontend), a district-by-district map of where open speech recognition fails in India.

## What's here

| Path | What it does |
|---|---|
| `atlas_eval/normalize.py` | Text normaliser used on references and hypotheses before scoring. The rules are in its docstring, with test pairs in `tests/normalize_pairs.tsv` |
| `atlas_eval/score.py` | `atlas-eval`: per-district WER and CER, speaker-bootstrap intervals, shrinkage for map colours; writes the JSON the site reads |
| `runners/` | One script per baseline. Each reads a manifest and writes raw transcripts. `uv run` gives each its own environment |
| `data/fleurs.py` | Builds a small FLEURS Hindi manifest for smoke tests |
| `data/vaani.py` | Indexes Vaani's files, cuts a test slice, and fetches the slice's audio. It reads only footers and the row groups it needs, because Vaani is tens of terabytes and the Hub allows 5,000 requests per five minutes |
| `data/splits.py` | Matches a manifest's clips to Vaani's official train, validation and test split (in `ARTPARK-IISc/Vaani-transcription-part`), and compares each model on train against held-out clips as a contamination check |
| `data/m1.py` | Surveys Vaani's official test split in every language, checks its images against train, and drafts the M1 test sets (`data/m1_test_survey.tsv`, `data/manifests/m1_{language}.jsonl`) |
| `data/benchmark.py` | Fetches a language of Vaani-Benchmark-V1.0 (held out, three transcriptions per segment) as a manifest, and scores it: the benchmark's own WER, our WER, transcriber disagreement per district, and, against M0, a contamination check |
| `data/vaani_districts.tsv` | Each Vaani district's polygons on the 2011 map; districts created since 2011 point to their parent |
| `data/m0_districts.tsv`, `data/manifests/m0.jsonl` | The ten M0 districts and their frozen test slice (no audio) |
| `runners/kaggle_m0.sh` | Runs M0 end to end on a Kaggle GPU: fetch audio, three baselines, score |
| `runners/kaggle_m1.sh` | Runs every M1 test set on Kaggle as a chain of 10-hour runs, each continuing from the last one's transcripts; instructions at its top |
| `runners/kaggle_bench.sh` | The same for Vaani-Benchmark-V1.0's Hindi set; then `uv run python data/benchmark.py score runs/bench/manifest.jsonl runs/bench/*.jsonl --m0 runs/m0` compares it with M0 |
| `audit/` | The M0 listening audit: picks 30 clips per district, builds a page for checking their transcripts by ear, and scores what the listener sends back |
| `notes/` | Decisions, results, and `m0_errors.tsv`: 50 errors per M0 model, sorted into the first error taxonomy |

## Formats

**Manifest** (JSONL, one clip per line): `id`, `audio` (path), `text` (reference), `language` (Vaani's language name), `district`, `state`, `speaker`.

**Transcripts** (JSONL, from a runner): `id`, `model`, `hyp`. `hyp` is `null` when the model doesn't support the clip's language, and the site shows that district as unsupported rather than scoring it.

## Smoke test

```sh
uv sync
uv run pytest
python data/fleurs.py runs/fleurs --n 50
uv run runners/whisper.py runs/fleurs/manifest.jsonl runs/fleurs/whisper.jsonl --model large-v3
uv run atlas-eval runs/fleurs/manifest.jsonl runs/fleurs/whisper.jsonl -o runs/fleurs/atlas.json
```

On a laptop CPU, use `--model tiny` to check the plumbing. Omnilingual ASR runs only on Linux, and IndicConformer needs its Hugging Face terms accepted first; run both on Kaggle or the GPU slice.

## M0 on Vaani

Accept Vaani's terms on Hugging Face and log in (`hf auth login`) first.

```sh
uv run --group data python data/vaani.py index Hindi runs/vaani/Hindi.json        # ~10 min, one request per file
uv run --group data python data/vaani.py slice runs/vaani/Hindi.json data/m0_districts.tsv -o data/manifests/m0.jsonl
```

The slice is committed, so you don't need to rebuild it. Manifests contain transcripts from [Vaani](https://vaani.iisc.ac.in) by IISc and ARTPARK (CC BY 4.0); speaker IDs are replaced by hashes, because some embed fragments of names. The baselines run on Kaggle; the instructions are at the top of `runners/kaggle_m0.sh`.

## M0 listening audit

Checks whether a district's high WER comes from its speakers or from its transcripts. `data/manifests/m0_audit.jsonl` holds 30 clips per district: up to 15 where all three models agree on a word the reference doesn't have, and the rest at random. It needs the three M0 transcript files in `runs/m0/`.

```sh
uv run --group data python data/vaani.py fetch data/manifests/m0_audit.jsonl runs/audit   # ~3.5 GB read, ~75 MB kept
uv run python audit/audit.py page                                                         # runs/audit/index.html
```

Zip `runs/audit/index.html` with `runs/audit/audio/` (not `manifest.jsonl`, which says which clips are suspects) and send it to a Hindi-speaking listener, who opens `index.html` in a browser and sends back `m0_audit_results.jsonl`. Answers are kept in their browser between sessions. Then:

```sh
uv run python audit/audit.py score m0_audit_results.jsonl   # reference WER per district, and each model's WER corrected for it
```

## M1 test sets

Drafts from Vaani's official test split, not frozen yet. Accept the terms of `ARTPARK-IISc/Vaani-transcription-part` first. Raw rows, which hold unhashed speaker IDs, are cached in `runs/vaani/`.

```sh
uv run --group data python data/m1.py survey -o data/m1_test_survey.tsv          # ~10 min the first time
uv run --group data python data/m1.py images Hindi Bengali                         # test images also in train
uv run --group data python data/m1.py slice Hindi -o data/manifests/m1_hindi.jsonl # one per language; prints its SHA-256
```

`vaani.py fetch` reads them from the repository named in each clip's `source.repo`.

## Rules

- Audio never goes into git. Commit manifests and scripts.
- Keep every raw transcript, so any number can be recomputed from them.
- No test speaker appears in any training data.
