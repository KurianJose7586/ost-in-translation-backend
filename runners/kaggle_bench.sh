#!/usr/bin/env bash
# Vaani-Benchmark-V1.0, Hindi, on Kaggle: fetch the segments, run the three baselines, score.
#
# First accept the benchmark's terms at https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Benchmark-V1.0
# with the account whose token is the notebook's HF_TOKEN secret. Set the notebook up as in
# runners/kaggle_m0.sh (accelerator "GPU T4 x2" is fastest), but end with:
#
#   !cd ost-in-translation-backend && git pull -q && bash runners/kaggle_bench.sh
#
# With two GPUs, Whisper and Omnilingual run side by side, one each. IndicConformer runs on
# the CPU after them: alongside, its ONNX threads starved the GPU models' CPU work.
#
# A draft session loses /kaggle/working when it stops, so for the full run delete every other
# cell, then use Save Version > Save & Run All: it runs in the background for up to 12 hours.
# Rerunning in the same session skips the clips already finished. Download runs/bench.zip
# (in the version's output) at the end.
set -uo pipefail
pip install -q uv
uv sync -q --group data

D=runs/bench
M=$D/manifest.jsonl
uv run --group data python data/benchmark.py fetch $D || exit 1

whisper() { CUDA_VISIBLE_DEVICES=$1 uv run runners/whisper.py $M $D/whisper.jsonl --device cuda || echo "whisper failed"; }
omni() { CUDA_VISIBLE_DEVICES=$1 uv run runners/omnilingual.py $M $D/omnilingual.jsonl --model omniASR_LLM_3B_v2 || echo "omnilingual failed"; }
if [ "$(nvidia-smi -L | wc -l)" -ge 2 ]; then
  whisper 0 &
  omni 1
  wait
else
  whisper 0
  omni 0
fi
uv run runners/indicconformer.py $M $D/indicconformer.jsonl || echo "indicconformer failed"

outs=$(ls $D/whisper.jsonl $D/omnilingual.jsonl $D/indicconformer.jsonl 2>/dev/null)
uv run python data/benchmark.py score $M $outs | tee $D/score.md
for f in $outs; do echo "$f: $(wc -l < "$f") of $(wc -l < "$M") segments"; done
(cd $D && zip -q ../bench.zip *.jsonl score.md)
echo "done: runs/bench.zip"
