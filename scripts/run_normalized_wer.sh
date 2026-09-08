#!/bin/bash
#SBATCH --job-name=emospp_normwer
#SBATCH --output=/data/user_data/xoy/slurm_logs/emospp_normwer_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/emospp_normwer_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --requeue
# Whisper-normalized WER for every results JSON (+ articulatory-tts transcript dumps
# for like-for-like floors/comparators). CPU-only: text normalization + jiwer.
set -euo pipefail
shopt -s nullglob
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
A=/data/user_data/xoy/articulatory-tts
EXTRA=("$A"/sparc_resynth_baselines/gt_transcripts_{ljspeech,libritts,libritts_test_other,esd}.json
       "$A"/sparc_resynth_baselines/resynth_transcripts_*.json
       "$A"/*largedim*/eval_*per_utt*.json)
echo "extra transcript files (${#EXTRA[@]}):"; printf '  %s\n' "${EXTRA[@]}"
"$EVAL_VENV/bin/python" -u eval/normalized_wer.py --extra "${EXTRA[@]}" --summary_md "$SLURM_SUBMIT_DIR/eval/results_wer_normalized.md"
echo "NORMWER COMPLETE $(date)"
