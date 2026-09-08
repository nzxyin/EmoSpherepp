#!/bin/bash
#SBATCH --job-name=emospp_synth
#SBATCH --output=/data/user_data/xoy/slurm_logs/emospp_synth_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/emospp_synth_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue

# Stage 2: EmoSphere++ synthesis for one test set -> wav_pairs/<dataset>/.
# Usage: sbatch scripts/run_synth.sh <ljspeech|libritts|libritts_test_other|esd> [extra synthesize.py args]
# Restart-safe (existing {uid}_pred.wav skipped via synth_log.jsonl).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
DS="${1:-}"; shift || true
[ -n "$DS" ] || { echo "usage: sbatch $0 <dataset>"; exit 2; }
echo "node=$(hostname) start=$(date) dataset=$DS extra=$*"
"$VENV/bin/python" eval/synthesize.py --dataset "$DS" "$@"
echo "SYNTH COMPLETE $DS $(date)"
