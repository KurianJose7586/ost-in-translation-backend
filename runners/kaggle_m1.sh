#!/usr/bin/env bash
# M1 on Kaggle: every M1 test set (data/manifests/m1_*.jsonl) through the three baselines, then atlas-eval.
#
# M1 is about 50 hours of audio, more than one 12-hour Kaggle run, so it runs as a chain:
# 1. A notebook with one cell, set up as in runners/kaggle_bench.sh (GPU T4 x2, internet,
#    HF_TOKEN secret, Vaani-transcription-part's terms accepted), ending with:
#      !cd ost-in-translation-backend && git pull -q && bash runners/kaggle_m1.sh
# 2. Save Version > Save & Run All. The models stop after HOURS (default 10), so the run ends
#    and Kaggle saves its output before the 12-hour limit.
# 3. For the next run, Add Input > the previous version's output, then Save & Run All again.
#    The script copies its transcripts and every runner skips the clips already done.
# Repeat until the last lines show every model at all clips. Download runs/m1.zip each time.
#
# IndicConformer runs on the CPU alongside the GPU models at the lowest priority, so its
# threads take only CPU time the others leave idle.
set -uo pipefail
pip install -q uv
uv sync -q --group data

D=runs/m1
M=$D/manifest.jsonl
END=$(( $(date +%s) + ${HOURS:-10} * 3600 ))
mkdir -p $D
cat data/manifests/m1_*.jsonl > $D/clips.jsonl
uv run --group data python data/vaani.py fetch $D/clips.jsonl $D || exit 1
for name in whisper omnilingual indicconformer; do
  f=$(find /kaggle/input -path "*/$D/$name.jsonl" 2>/dev/null | head -1)
  [ -n "$f" ] && [ ! -f $D/$name.jsonl ] && cp "$f" $D/ && echo "resuming from $f"
done

left() { n=$(( END - $(date +%s) )); echo $(( n > 1 ? n : 1 )); }  # timeout 0 would mean no limit
whisper() { CUDA_VISIBLE_DEVICES=$1 timeout "$(left)" uv run runners/whisper.py $M $D/whisper.jsonl --device cuda || echo "whisper stopped"; }
omni() { CUDA_VISIBLE_DEVICES=$1 timeout "$(left)" uv run runners/omnilingual.py $M $D/omnilingual.jsonl --model omniASR_LLM_3B_v2 || echo "omnilingual stopped"; }
(timeout "$(left)" nice -n 19 uv run runners/indicconformer.py $M $D/indicconformer.jsonl || echo "indicconformer stopped") &
if [ "$(nvidia-smi -L | wc -l)" -ge 2 ]; then
  whisper 0 &
  omni 1
else
  whisper 0
  omni 0
fi
wait

outs=$(ls $D/whisper.jsonl $D/omnilingual.jsonl $D/indicconformer.jsonl 2>/dev/null)
uv run atlas-eval "$M" $outs -o $D/atlas.json --round m1
for f in $outs; do echo "$f: $(wc -l < "$f") of $(wc -l < "$M") clips"; done
(cd $D && zip -q ../m1.zip whisper.jsonl omnilingual.jsonl indicconformer.jsonl atlas.json 2>/dev/null)
rm -rf $D/audio  # the next run fetches it again; this keeps the saved output small
echo "done: runs/m1.zip"
