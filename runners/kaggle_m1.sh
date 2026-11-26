#!/usr/bin/env bash
# M1 on Kaggle: M1 test sets (data/manifests/m1_*.jsonl) through the three baselines, then atlas-eval.
#
# Hindi first: it alone decides the worst-20 set, so by default only m1_hindi.jsonl runs.
# LANGS='*' runs every language; the transcripts carry over, so Hindi isn't redone.
#
# M1 is about 50 hours of audio, more than one 12-hour Kaggle run, so it runs as a chain:
# 1. A notebook with one cell, set up as in runners/kaggle_bench.sh (GPU T4 x2, internet,
#    HF_TOKEN secret, Vaani-transcription-part's terms accepted), ending with:
#      !cd ost-in-translation-backend && git pull -q && bash runners/kaggle_m1.sh
#    (or `LANGS='*' bash runners/kaggle_m1.sh` once Hindi is done).
# 2. Save Version > Save & Run All. The models stop after HOURS (default 10), so the run ends
#    and Kaggle saves its output before the 12-hour limit.
# 3. For the next run, Add Input > the previous version's output, then Save & Run All again.
#    The script copies its transcripts, fetches audio only for clips some model still lacks
#    (never for languages no model supports), and every runner skips the clips already done.
# Repeat until the last lines show every model at all clips. Download runs/m1.zip each time.
# Until then atlas.json is a progress round, marked partial.
#
# IndicConformer runs on the CPU alongside the GPU models at the lowest priority, so its
# threads take only CPU time the others leave idle. With accelerator "None", only IndicConformer
# runs, and the run uses none of the weekly GPU hours.
set -uo pipefail
pip install -q uv
uv sync -q --group data

D=runs/m1
M=$D/manifest.jsonl
END=$(( $(date +%s) + ${HOURS:-10} * 3600 ))
mkdir -p $D
cat data/manifests/m1_${LANGS:-hindi}.jsonl > $D/clips.jsonl || exit 1
for name in whisper omnilingual indicconformer; do
  f=$(find /kaggle/input -path "*/$D/$name.jsonl" 2>/dev/null | head -1)
  [ -n "$f" ] && [ ! -f $D/$name.jsonl ] && cp "$f" $D/ && echo "resuming from $f"
done
uv run python runners/resume.py $D/clips.jsonl $D > $D/todo.jsonl
echo "$(wc -l < $D/todo.jsonl) of $(wc -l < $D/clips.jsonl) clips still need audio"
: > $M
if [ -s $D/todo.jsonl ]; then uv run --group data python data/vaani.py fetch $D/todo.jsonl $D || exit 1; fi
uv run python runners/resume.py $D/clips.jsonl $D --no-audio >> $M  # unsupported languages: no audio needed
if [ -s $M ]; then

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
fi

outs=$(ls $D/whisper.jsonl $D/omnilingual.jsonl $D/indicconformer.jsonl 2>/dev/null)
uv run atlas-eval $D/clips.jsonl $outs -o $D/atlas.json --round m1 --allow-partial
for f in $outs; do echo "$f: $(wc -l < "$f") of $(wc -l < $D/clips.jsonl) clips"; done
(cd $D && zip -q ../m1.zip whisper.jsonl omnilingual.jsonl indicconformer.jsonl atlas.json 2>/dev/null)
rm -rf $D/audio  # the next run fetches it again; this keeps the saved output small
echo "done: runs/m1.zip"
