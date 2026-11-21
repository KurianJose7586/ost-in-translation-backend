#!/usr/bin/env bash
# M0 on Kaggle (GPU notebook): fetch the slice's audio, run the three baselines, score.
#
# In a Kaggle notebook with a GPU and internet on, add your Hugging Face token as the secret HF_TOKEN, then run:
#
#   import os; from kaggle_secrets import UserSecretsClient
#   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
#   !test -d ost-in-translation-backend || git clone -q https://github.com/KurianJose7586/ost-in-translation-backend.git
#   !cd ost-in-translation-backend && git pull -q && bash runners/kaggle_m0.sh
#
# Safe to rerun after a crash or a lost session: every step skips the clips it already finished.
# Outputs land in runs/m0/: raw transcripts per model, and atlas.json. Download runs/m0/*.jsonl and atlas.json.
set -uo pipefail
pip install -q uv
uv sync -q --group data

M=runs/m0/manifest.jsonl
uv run --group data python data/vaani.py fetch data/manifests/m0.jsonl runs/m0 || exit 1

# One model failing shouldn't lose the others' transcripts.
uv run runners/whisper.py "$M" runs/m0/whisper.jsonl --device cuda || echo "whisper failed"
uv run runners/omnilingual.py "$M" runs/m0/omnilingual.jsonl --model omniASR_LLM_3B_v2 || echo "omnilingual failed"
uv run runners/indicconformer.py "$M" runs/m0/indicconformer.jsonl || echo "indicconformer failed"

outs=$(ls runs/m0/whisper.jsonl runs/m0/omnilingual.jsonl runs/m0/indicconformer.jsonl 2>/dev/null)
uv run atlas-eval "$M" $outs -o runs/m0/atlas.json --round m0
for f in $outs; do echo "$f: $(wc -l < "$f") of $(wc -l < "$M") clips"; done
echo "done"
