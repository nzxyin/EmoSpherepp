#!/bin/bash
#SBATCH --job-name=emospp_setup
#SBATCH --output=/data/user_data/xoy/slurm_logs/emospp_setup_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/emospp_setup_%j.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue

# One-shot environment bootstrap + smoke test for the EmoSphere++ eval (2026-09-07).
# Everything that touches /data has to run on a compute node. Steps:
#   1. uv venv on /data (symlinked to ./.venv), deps from pyproject.toml/uv.lock
#   2. released checkpoint (Works Drive) + vocoder/feature-model downloads
#   3. sanity: data roots/split files, GPU, module imports, checkpoint keys
#   4. phone_set.json rebuilt from esd_text_emo.txt and checked against the ckpt vocab
#   5. manifests for the 4 test sets (+ esd_all), then a 3-utterance end-to-end smoke synth
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
echo "node=$(hostname) date=$(date)"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

echo "=== 1. venv ($VENV) ==="
if [ ! -x "$VENV/bin/python" ]; then uv venv "$VENV" --python 3.10; fi
if [ -f uv.lock ]; then uv sync --frozen --no-install-project 2>&1 | tail -20; else uv sync --no-install-project 2>&1 | tail -20; fi
ln -sfn "$VENV" .venv
PY="$VENV/bin/python"
"$PY" -c "import torch, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

echo "=== 2. downloads ==="
bash scripts/download_ckpt.sh "$EVAL_ROOT/ckpt/model_ckpt_steps_1100000.ckpt"
"$PY" - <<'PYEOF'
import os, torch
ck = torch.load(os.environ["EVAL_ROOT"] + "/ckpt/model_ckpt_steps_1100000.ckpt", map_location="cpu", weights_only=False)
print("ckpt top-level keys:", list(ck.keys()), "global_step:", ck.get("global_step"), "epoch:", ck.get("epoch"))
sd = ck["state_dict"]
if "model" in sd and isinstance(sd["model"], dict): sd = sd["model"]
print(f"{len(sd)} tensors; sample keys:", list(sd.keys())[:5])
for k, v in sd.items():
    if any(s in k for s in ("encoder.emb.weight", "proj_m.weight", "spk_mlp.0.weight", "emo_mlp.0.weight", "emo_VAD_inten_proj", "azimuth_emb", "elevation_emb")):
        print(f"  {k}: {tuple(v.shape)}")
PYEOF
echo "--- any locally available 16k BigVGAN? (informational) ---"
timeout 120 find /data/user_data/xoy -maxdepth 4 \( -iname "*bigvgan*" -o -iname "bigV_16k*" -o -iname "g_0*0" \) 2>/dev/null | head -20 || true
"$PY" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
for repo in ["speechbrain/tts-hifigan-libritts-16kHz", "microsoft/wavlm-base-sv", "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", "emotion2vec/emotion2vec_plus_base"]:
    p = snapshot_download(repo); print("cached:", repo, "->", p)
PYEOF

echo "=== 3. data sanity ==="
for p in /data/user_data/xoy/LJSpeech-1.1/preprocessed/test.json /data/user_data/xoy/LJSpeech-1.1/metadata.csv /data/user_data/xoy/LJSpeech-1.1/wavs \
         /data/user_data/xoy/LibriTTS_R/test-clean.json /data/user_data/xoy/LibriTTS_R/test-clean \
         /data/user_data/xoy/LibriTTS_R/test-other.json /data/group_data/UTD-NAS/Databases/LibriTTS-R/LibriTTS_R/test-other \
         /data/user_data/xoy/esd_english_splits/test.tsv /data/group_data/UTD-NAS/Databases/ESD/ESD \
         /data/user_data/xoy/venvs/eval-articulatory-tts/bin/python /data/user_data/xoy/venvs/eval-emotion/bin/python; do
  if [ -e "$p" ]; then echo "OK  $p"; else echo "MISSING $p"; fi
done
"$PY" -c "
import json,csv
print('ljspeech test n=', len(json.load(open('/data/user_data/xoy/LJSpeech-1.1/preprocessed/test.json'))))
print('libritts test-clean n=', len(json.load(open('/data/user_data/xoy/LibriTTS_R/test-clean.json'))))
print('libritts test-other n=', len(json.load(open('/data/user_data/xoy/LibriTTS_R/test-other.json'))))
rows=list(csv.DictReader(open('/data/user_data/xoy/esd_english_splits/test.tsv'), delimiter='\t'))
print('esd test n=', len(rows), 'columns=', list(rows[0].keys()), 'first=', rows[0])
import collections; print('esd test per-speaker:', dict(collections.Counter(r['stem'].split('_')[0] for r in rows)))
print('esd test stems for 0011 (first 5 sorted):', sorted(r['stem'] for r in rows if r['stem'].startswith('0011'))[:5])
"
ls /data/group_data/UTD-NAS/Databases/ESD/ESD | head -5; ls /data/group_data/UTD-NAS/Databases/ESD/ESD/0011 | head; ls /data/group_data/UTD-NAS/Databases/ESD/ESD/0011/Angry | head -3

echo "=== 4. imports + phone set ==="
"$PY" -c "import diffusers, conformer, einops, speechbrain, funasr, g2p_en, librosa, pyloudnorm; print('imports OK: diffusers', diffusers.__version__)"
"$PY" -c "from models.tts.EmoSpherepp import EmoSpherepp; from models.tts.EmoSpherepp.transformer import BasicTransformerBlock; print('model modules import OK')"
"$PY" eval/build_phone_set.py

echo "=== 5. manifests + smoke test ==="
"$PY" eval/prepare_manifests.py
"$PY" eval/extract_ref_features.py --manifest ljspeech --limit 3
"$PY" eval/extract_ref_features.py --manifest esd --limit 3
"$PY" eval/synthesize.py --dataset ljspeech --limit 3 --out_dir "$EVAL_ROOT/wav_pairs/smoke_ljspeech"
"$PY" eval/synthesize.py --dataset ljspeech --limit 3 --resynth --out_dir "$EVAL_ROOT/wav_pairs/smoke_ljspeech_resynth"
ls -la "$EVAL_ROOT/wav_pairs/smoke_ljspeech"; cat "$EVAL_ROOT/wav_pairs/smoke_ljspeech/synth_log.jsonl"
echo "=== smoke score (eval venv) ==="
"$EVAL_VENV/bin/python" eval/score_wav_pairs.py --dataset ljspeech --wav_pairs_dir "$EVAL_ROOT/wav_pairs/smoke_ljspeech" --results_path "$EVAL_ROOT/results/smoke_ljspeech.json" --per_utt_out "$EVAL_ROOT/results/smoke_ljspeech_per_utt.json"
"$EVAL_VENV/bin/python" eval/score_wav_pairs.py --dataset ljspeech --wav_pairs_dir "$EVAL_ROOT/wav_pairs/smoke_ljspeech_resynth" --results_path "$EVAL_ROOT/results/smoke_ljspeech_resynth.json"
cat "$EVAL_ROOT/results/smoke_ljspeech_per_utt.json"
echo "SETUP COMPLETE $(date)"
