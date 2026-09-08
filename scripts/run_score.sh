#!/bin/bash
#SBATCH --job-name=emospp_score
#SBATCH --output=/data/user_data/xoy/slurm_logs/emospp_score_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/emospp_score_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue

# Stage 3: score wav_pairs/<tag>/ with articulatory-tts's metric stack.
#   base metrics (WER / UTMOSv2 / DNSMOS / speaker_cosine): eval/score_wav_pairs.py
#       under the eval-articulatory-tts venv (same models as eval_full_testset.py)
#   emotion_cosine: articulatory-tts/score_side_metric.py --metric emotion, unchanged,
#       under the eval-emotion venv; merged with articulatory-tts/merge_eval_results.py
# Usage: sbatch scripts/run_score.sh <dataset> [tag]   (tag defaults to <dataset>; e.g. esd_resynth)
# Stage-skip resume: each output JSON is skipped when it already exists.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
DS="${1:-}"; TAG="${2:-$DS}"
[ -n "$DS" ] || { echo "usage: sbatch $0 <dataset> [tag]"; exit 2; }
WAV_PAIRS="$EVAL_ROOT/wav_pairs/$TAG"
OUT="$EVAL_ROOT/results"; mkdir -p "$OUT"
echo "node=$(hostname) start=$(date) dataset=$DS tag=$TAG pairs=$WAV_PAIRS"
BASE="$OUT/eval_${TAG}.json"
if [ -s "$BASE" ]; then echo "skip base metrics (exists: $BASE)"; else
  "$EVAL_VENV/bin/python" -u eval/score_wav_pairs.py --dataset "$DS" --wav_pairs_dir "$WAV_PAIRS" \
    --results_path "$BASE" --per_utt_out "$OUT/eval_${TAG}_per_utt.json"
fi
EMO="$OUT/eval_${TAG}_emotion.json"
if [ -s "$EMO" ]; then echo "skip emotion_cosine (exists: $EMO)"; else
  "$EMOTION_VENV/bin/python" -u "$ARTIC_TTS/score_side_metric.py" --metric emotion \
    --wav_pairs_dir "$WAV_PAIRS" --results_path "$EMO"
fi
if grep -q '"emotion_cosine"' "$BASE"; then echo "emotion_cosine already merged"; else
  "$EVAL_VENV/bin/python" -u "$ARTIC_TTS/merge_eval_results.py" --base "$BASE" --side "$EMO"
fi
echo "SCORE COMPLETE $TAG $(date)"
