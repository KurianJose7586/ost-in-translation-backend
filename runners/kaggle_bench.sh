#!/usr/bin/env bash
# Vaani-Benchmark-V1.0, Hindi, on Kaggle: fetch the segments, run the three baselines, score.
#
# First accept the benchmark's terms at https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Benchmark-V1.0
# with the account whose token is the notebook's HF_TOKEN secret. Set the notebook up as in
# runners/kaggle_m0.sh (accelerator "GPU T4 x2"), with one cell ending:
#
#   !cd ost-in-translation-backend && git pull -q && bash runners/kaggle_bench.sh
#
# Then Save Version > Save & Run All. Omnilingual takes about 12 seconds a segment on a T4, more
# than one 12-hour Kaggle run, so the models stop after HOURS (default 10) and the run scores and
# saves what is done. For the next run, Add Input > the previous version's output, then Save &
# Run All again: the script copies its transcripts and every runner skips the segments already
# done. Repeat until the last lines show every model at all segments. Download runs/bench.zip
# each time.
#
# IndicConformer runs on the CPU alongside the GPU models at the lowest priority, so its
# threads take only CPU time the others leave idle. With accelerator "None", only IndicConformer
# runs, and the run uses none of the weekly GPU hours.
set -uo pipefail
pip install -q uv
uv sync -q --group data

D=runs/bench
M=$D/manifest.jsonl
END=$(( $(date +%s) + ${HOURS:-10} * 3600 ))
uv run --group data python data/benchmark.py fetch $D || exit 1
for name in whisper omnilingual indicconformer; do
  f=$(find /kaggle/input -path "*/$D/$name.jsonl" 2>/dev/null | head -1)
  [ -n "$f" ] && [ ! -f $D/$name.jsonl ] && cp "$f" $D/ && echo "resuming from $f"
done

left() { n=$(( END - $(date +%s) )); echo $(( n > 1 ? n : 1 )); }  # timeout 0 would mean no limit
whisper() { CUDA_VISIBLE_DEVICES=$1 timeout "$(left)" uv run runners/whisper.py $M $D/whisper.jsonl --device cuda || echo "whisper stopped"; }
omni() { CUDA_VISIBLE_DEVICES=$1 timeout "$(left)" uv run runners/omnilingual.py $M $D/omnilingual.jsonl --model omniASR_LLM_3B_v2 || echo "omnilingual stopped"; }
(timeout "$(left)" nice -n 19 uv run runners/indicconformer.py $M $D/indicconformer.jsonl || echo "indicconformer stopped") &
gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$gpus" -ge 2 ]; then
  whisper 0 &
  omni 1
elif [ "$gpus" -eq 1 ]; then
  whisper 0
  omni 0
fi  # no GPU: only IndicConformer runs
wait

outs=$(ls $D/whisper.jsonl $D/omnilingual.jsonl $D/indicconformer.jsonl 2>/dev/null)
uv run python data/benchmark.py score $M $outs | tee $D/score.md
for f in $outs; do echo "$f: $(wc -l < "$f") of $(wc -l < "$M") segments"; done
(cd $D && zip -q ../bench.zip *.jsonl score.md)
rm -rf $D/audio  # the next run fetches it again; this keeps the saved output small
echo "done: runs/bench.zip"
