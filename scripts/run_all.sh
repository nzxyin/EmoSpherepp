#!/bin/bash
# Launch the whole EmoSphere++ evaluation as a SLURM dependency chain (run on the
# login node after scripts/setup_env.sh has succeeded):
#   features (all sets)  ->  synth <ds>  ->  score <ds>          for each test set (parallel)
#                        ->  synth --resynth <ds> -> score <ds>_resynth   (vocoder-ceiling anchors)
# Usage: scripts/run_all.sh [datasets...]   (default: esd ljspeech libritts libritts_test_other)
#        RESYNTH="esd ljspeech" scripts/run_all.sh     # which sets also get a resynthesis anchor
set -euo pipefail
cd "$(dirname "$0")/.."
DATASETS=("$@"); [ ${#DATASETS[@]} -gt 0 ] || DATASETS=(esd ljspeech libritts libritts_test_other)
RESYNTH=${RESYNTH:-"esd ljspeech"}
FEAT=$(sbatch --parsable scripts/run_features.sh)
echo "features job: $FEAT"
for ds in "${DATASETS[@]}"; do
  S=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$FEAT --job-name="emospp_synth_$ds" scripts/run_synth.sh "$ds")
  C=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$S --job-name="emospp_score_$ds" scripts/run_score.sh "$ds")
  echo "$ds: synth $S -> score $C"
  if [[ " $RESYNTH " == *" $ds "* ]]; then
    RS=$(sbatch --parsable --job-name="emospp_resynth_$ds" scripts/run_synth.sh "$ds" --resynth)  # anchors read no features
    RC=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$RS --job-name="emospp_score_${ds}_resynth" scripts/run_score.sh "$ds" "${ds}_resynth")
    echo "$ds resynth anchor: synth $RS -> score $RC"
  fi
done
squeue -u "$USER" -n "$(printf 'emospp_%s,' features synth_esd synth_ljspeech synth_libritts synth_libritts_test_other)" -o "%i %j %T %R" 2>/dev/null || true
