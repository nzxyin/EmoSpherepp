#!/bin/bash
#SBATCH --job-name=emospp_breakdown
#SBATCH --output=/data/user_data/xoy/slurm_logs/emospp_breakdown_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/emospp_breakdown_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --exclude=babel-l9-16
# Per-utterance emotion cosine (eval-emotion venv) + per-emotion / seen-unseen breakdown
# (eval-articulatory-tts venv) for ESD outputs and the ESD vocoder anchor.
# Usage: sbatch scripts/run_breakdown.sh [tags...]   (default: esd esd_resynth)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
TAGS=("$@"); [ ${#TAGS[@]} -gt 0 ] || TAGS=(esd esd_resynth)
for tag in "${TAGS[@]}"; do
  "$EMOTION_VENV/bin/python" -u eval/emotion_cosine_per_utt.py --wav_pairs_dir "$EVAL_ROOT/wav_pairs/$tag" \
      --out "$EVAL_ROOT/results/eval_${tag}_emotion_per_utt.json"
  "$EVAL_VENV/bin/python" -u eval/breakdown.py --tag "$tag" --dataset esd --out_md "$SLURM_SUBMIT_DIR/eval/results_${tag}_breakdown.md"
done
echo "BREAKDOWN COMPLETE $(date)"
