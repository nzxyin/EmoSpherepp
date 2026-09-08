#!/bin/bash
#SBATCH --job-name=emospp_features
#SBATCH --output=/data/user_data/xoy/slurm_logs/emospp_features_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/emospp_features_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue

# Stage 1: reference-audio features for every test utterance (WavLM x-vector,
# emotion2vec+ base, audeering VAD) + VAD over ALL 17.5k ESD English utterances
# (VAD only) to reproduce the per-emotion intensity-normalization quartiles.
# Restart-safe: every utterance's .pt is skipped once present, so a preempt
# requeue just resumes.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
PY="$VENV/bin/python"
echo "node=$(hostname) start=$(date)"
"$PY" eval/extract_ref_features.py --manifest esd_all --features vad --max_errors 100
"$PY" eval/extract_ref_features.py --esd_stats
for ds in esd ljspeech libritts libritts_test_other; do
  "$PY" eval/extract_ref_features.py --manifest "$ds" --features wavlm,e2v,vad
done
echo "FEATURES COMPLETE $(date)"
