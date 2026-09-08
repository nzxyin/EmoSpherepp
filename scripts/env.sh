# Shared environment for every EmoSphere++ eval job on Babel. Source from sbatch
# scripts AFTER `cd "$SLURM_SUBMIT_DIR"`.
export PATH="$HOME/.local/bin:$PATH"
export EVAL_ROOT=/data/user_data/xoy/EmoSpherepp_eval
export VENV=/data/user_data/xoy/venvs/emospherepp
# IMPORTANT: ~/.zshrc exports UV_PROJECT_ENVIRONMENT pointing at the
# articulatory-tts venv and sbatch inherits the submitting shell's env -- an
# unqualified `uv sync` here would clobber that venv. Always pin ours.
export UV_PROJECT_ENVIRONMENT=$VENV
export UV_CACHE_DIR=/data/user_data/xoy/.cache/uv
# Caches off $HOME. Same HF_HOME as articulatory-tts's eval jobs (speechbrain
# savedirs live under it); hub downloads go to the shared group cache.
export HF_HOME=/data/user_data/xoy/.cache/huggingface
export HF_HUB_CACHE=/data/hf_cache/hub
export XDG_CACHE_HOME=/data/user_data/xoy/.cache
export TORCH_HOME=/data/user_data/xoy/.cache/torch
export NLTK_DATA=/data/user_data/xoy/.cache/nltk_data
export MPLCONFIGDIR=/data/user_data/xoy/.cache/matplotlib
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
# The two articulatory-tts eval venvs we reuse for scoring (identical metric stack).
export EVAL_VENV=/data/user_data/xoy/venvs/eval-articulatory-tts
export EMOTION_VENV=/data/user_data/xoy/venvs/eval-emotion
export ARTIC_TTS=/home/xoy/articulatory-tts
mkdir -p "$EVAL_ROOT" "$NLTK_DATA" "$MPLCONFIGDIR" /data/user_data/xoy/slurm_logs
