# Lost in Transcription: backend

Evaluation, data and (later) training code for [Lost in Transcription](https://github.com/KurianJose7586/lost-in-translation-frontend), a district-by-district map of where open speech recognition fails in India.

## What's here

| Path | What it does |
|---|---|
| `atlas_eval/normalize.py` | Text normaliser used on references and hypotheses before scoring. The rules are in its docstring, with test pairs in `tests/normalize_pairs.tsv` |
| `atlas_eval/score.py` | `atlas-eval`: WER and CER per district and language, speaker-bootstrap intervals, shrinkage for map colours, split-half reliability; writes the JSON the site reads, and refuses to compare models on different clips |
| `runners/` | One script per baseline. Each reads a manifest and writes raw transcripts. `uv run` gives each its own environment. `runners/resume.py` lets a run resume, and lists the clips a chained run still needs audio for |
| `results/m0/` | M0's raw transcripts, from which `atlas-eval` rebuilds the published round byte for byte |
| `data/fleurs.py` | Builds a small FLEURS Hindi manifest for smoke tests |
| `data/vaani.py` | Indexes Vaani's files, cuts a test slice, and fetches the slice's audio. It reads only footers and the row groups it needs, because Vaani is tens of terabytes and the Hub allows 5,000 requests per five minutes |
| `data/splits.py` | Matches a manifest's clips to Vaani's official train, validation and test split (in `ARTPARK-IISc/Vaani-transcription-part`), and compares each model on train against held-out clips as a contamination check |
| `data/m1.py` | Surveys Vaani's official test split in every language, checks its images against train, and drafts the M1 test sets (`data/m1_test_survey.tsv`, `data/manifests/m1_{language}.jsonl`) |
| `data/benchmark.py` | Fetches a language of Vaani-Benchmark-V1.0 (held out, three transcriptions per segment) as a manifest, and scores it: the benchmark's own WER, our WER, transcriber disagreement per district, and, against M0, a contamination check |
| `data/vaani_districts.tsv` | Each Vaani district's polygons on the 2011 map; districts created since 2011 point to their parent |
| `data/m0_districts.tsv`, `data/manifests/m0.jsonl` | The ten M0 districts and their frozen test slice (no audio) |
| `runners/kaggle_m0.sh` | Runs M0 end to end on a Kaggle GPU: fetch audio, three baselines, score |
| `runners/kaggle_m1.sh` | Runs M1 on Kaggle, Hindi first, as a chain of 10-hour runs, each continuing from the last one's transcripts and fetching only the audio still needed; instructions at its top |
| `runners/kaggle_bench.sh` | The same for Vaani-Benchmark-V1.0's Hindi set; then `uv run python data/benchmark.py score runs/bench/manifest.jsonl runs/bench/*.jsonl --m0 results/m0 --retest ../frontend/data/baseline.json` compares it with M0 |
| `audit/` | The M0 listening audit: picks 30 clips per district, builds a page for checking their transcripts by ear, and scores what the listener sends back |
| `notes/` | Decisions, results, and `m0_errors.tsv`: 50 errors per M0 model, sorted into the first error taxonomy |

## Formats

**Manifest** (JSONL, one clip per line): `id`, `audio` (path), `text` (reference), `language` (Vaani's language name), `district`, `state`, `speaker`.

**Transcripts** (JSONL, from a runner): `id`, `model`, `hyp`. `hyp` is `null` when the model doesn't support the clip's language, and the site shows that district as unsupported rather than scoring it. A clip the model couldn't read has `hyp: ""` and an `error`; `atlas-eval` leaves it out for every model and says how many. To retry those clips, delete their lines and rerun.

**Round** (JSON, from `atlas-eval`, version 2): `round`, `models`, `languages` (most clips first), `reliability` (split-half, per language and model), and `districts`, each with `id`, `name`, `state`, `map` (2011 polygons), `languages` (each one's share of the district's clips), `results[language][model]` and `examples[language]`. A result is either `{"status": "unsupported"}` or `wer`, `ci`, `wer_lenient`, `ci_lenient`, `wer_pooled` (map colour), `cer`, `clips` and `speakers`. `"partial": true` marks a round in which some model lacks some clips.

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

Checks whether a district's high WER comes from its speakers or from its transcripts. `data/manifests/m0_audit.jsonl` holds 30 clips per district: up to 15 where all three models agree on a word the reference doesn't have, and the rest at random. It reads the M0 transcripts in `results/m0/`.

```sh
uv run --group data python data/vaani.py fetch data/manifests/m0_audit.jsonl runs/audit   # ~3.5 GB read, ~75 MB kept
uv run python audit/audit.py page                                                         # runs/audit/index.html
```

Zip `runs/audit/index.html` with `runs/audit/audio/` (not `manifest.jsonl`, which says which clips are suspects) and send it to a Hindi-speaking listener, who opens `index.html` in a browser and sends back `m0_audit_results.jsonl`. Answers are kept in their browser between sessions. Then:

```sh
uv run python audit/audit.py score m0_audit_results.jsonl   # reference WER per district, and each model's WER corrected for it
```

Better, use two listeners and have both do the same district's 30 clips, splitting the rest. `audit.py agree a.jsonl b.jsonl` then says how far two careful listeners differ, which is the floor under any district's reference WER. Concatenate their files for `score`, keeping one listener's answers for the shared district.

## M1 test sets

Drafts from Vaani's official test split, not frozen yet. Accept the terms of `ARTPARK-IISc/Vaani-transcription-part` first. Raw rows, which hold unhashed speaker IDs, are cached in `runs/vaani/`.

```sh
uv run --group data python data/m1.py survey -o data/m1_test_survey.tsv          # ~10 min the first time
uv run --group data python data/m1.py images Hindi Bengali                         # test images also in train
uv run --group data python data/m1.py slice Hindi -o data/manifests/m1_hindi.jsonl # one per language; prints its SHA-256
```

`vaani.py fetch` reads them from the repository named in each clip's `source.repo`. Each clip's `image_in_train` says whether train clips describe the same image; `m1.py images` also says how many train clips describe an image the draft uses.

## Omnilingual in float16

M0 and the benchmark ran Omnilingual in bfloat16, which a T4 doesn't support natively (7–12 s a clip; M1 would need 55–95 T4-hours). Before M1 uses float16, check it gives the same WER on 100 M0 clips. In a Kaggle notebook set up as in `runners/kaggle_m0.sh`:

```sh
head -100 data/manifests/m0.jsonl > runs/fp16.jsonl
uv run --group data python data/vaani.py fetch runs/fp16.jsonl runs/fp16
uv run runners/omnilingual.py runs/fp16/manifest.jsonl runs/fp16/omnilingual.jsonl --dtype float16
```

Its transcripts carry the model id `omniASR_LLM_3B_v2-float16`, so they can't mix with bfloat16 ones. Locally, `uv run atlas-eval runs/fp16/manifest.jsonl runs/fp16/omnilingual.jsonl results/m0/omnilingual.jsonl -o runs/fp16/atlas.json` puts both on the same clips; compare their WER, and the log's seconds per clip. If they match, add `--dtype float16` to the `omni` line in `runners/kaggle_m1.sh`, in a fresh output directory.

## Rules

- Audio never goes into git. Commit manifests and scripts.
- Keep every raw transcript, so any number can be recomputed from them.
- No test speaker appears in any training data.
