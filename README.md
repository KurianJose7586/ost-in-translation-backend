# Lost in Transcription: backend

Evaluation, data and (later) training code for [Lost in Transcription](https://github.com/KurianJose7586/lost-in-translation-frontend), a district-by-district map of where open speech recognition fails in India.

## What's here

| Path | What it does |
|---|---|
| `atlas_eval/normalize.py` | Text normaliser used on references and hypotheses before scoring. The rules are in its docstring, with test pairs in `tests/normalize_pairs.tsv` |
| `atlas_eval/score.py` | `atlas-eval`: per-district WER and CER, speaker-bootstrap intervals, shrinkage for map colours; writes the JSON the site reads |
| `runners/` | One script per baseline. Each reads a manifest and writes raw transcripts. `uv run` gives each its own environment |
| `data/fleurs.py` | Builds a small FLEURS Hindi manifest for smoke tests |

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

On a laptop CPU, use `--model tiny` to check the plumbing. Omnilingual ASR runs only on Linux, and IndicConformer needs its Hugging Face terms accepted first; run both on Kaggle or the DGX.

## Rules

- Audio never goes into git. Commit manifests and scripts.
- Keep every raw transcript, so any number can be recomputed from them.
- No test speaker appears in any training data.
