# Changelog

All changes to this local clone of Choddeok/EmoSpherepp. Results and current
direction live in [CLAUDE.md](CLAUDE.md); this file is the chronological record.

## 2026-09-08

- Whisper-normalized WER (`wer_whisper_normalized`, EnglishTextNormalizer via
  `WhisperTokenizer.normalize` on both sides, empty normalized refs skipped —
  identical to articulatory-tts's GH #32 fix and the TTS repo's extra key) added
  to every results JSON by `eval/normalized_wer.py` (CPU job
  `scripts/run_normalized_wer.sh`, 10357908) from the stored transcripts, with
  the articulatory-tts GT/SPARC-resynthesis floors re-scored the same way
  (`eval/results_wer_normalized.md`). `score_wav_pairs.py` now records the
  normalized text and the metric inline for future runs; raw `wer` kept for
  comparability with the pre-fix articulatory-tts tables.
- Forked upstream to github.com/nzxyin/EmoSpherepp (`origin`; the original is
  `upstream`) and pushed this work as four commits: lazy `monotonic_align` import,
  uv environment + checkpoint download, inference/evaluation pipeline, docs.
- Evaluation complete: ESD scoring (10353878) and ESD vocoder anchor (10353879)
  finished 01:20 after the `--exclude=babel-l9-16` rerun; LJSpeech vocoder-native
  anchor (10354031/10354032) done 01:09. Final numbers in CLAUDE.md "Status /
  results". Foothold job 10353867 cancelled.
- Adversarial review of the pipeline (5-lens Workflow, 3 refuters per finding; 11
  confirmed, none affecting computed numbers). Fixes: `text_to_tokens` truncates to
  `max_input_tokens` (1550) before intersperse like `BaseSpeechDataset`;
  `read_esd_speaker_txt` tolerates a missing per-speaker txt (WER-only degradation,
  as upstream); results JSON `split` now = split-file name (eval_full_testset.py
  semantics) with the wav-pairs tag under `tag`, and `n_utterances` = split size
  from the new `manifests/<ds>.stats.json`; `run_all.sh` submits dependents with
  `--kill-on-invalid-dep=yes` and no longer gates the resynthesis anchors on the
  features job. New `--mel_center speechbrain` anchor mode (vocoder-native
  center=True mel, dir suffix `_resynth_native`) to measure the half-hop framing
  difference between EmoSphere++'s mel convention and the vocoder's; LJSpeech run
  submitted (jobs 10354031/10354032). Documented-only deviations: VAD quartiles
  include the few utterances the binarizer would have dropped for empty F0;
  `spherical_vector` clips `acos` input to [-1, 1] (NaN guard the original lacks).
- Scorer hardening after the first full run: per-utterance JSONL checkpoint/resume
  (`--partial_path`, default `<results>.partial.jsonl`), tolerant parsing of a
  truncated checkpoint line, per-utterance error isolation, atomic
  (`tmp` + `os.replace`) writes of the results and per-utterance JSONs, and a
  manifest-vs-pairs count check recorded as `n_utterances`/`n_pairs`/`n_predicted`.
- `synthesize.py` exits non-zero if any utterance failed or lacks a `_pred.wav`
  (so `afterok` scoring dependencies never run on an incomplete set) and
  tolerates a truncated `synth_log.jsonl` line; `extract_ref_features.py` gained
  `--max_errors` (0 for test sets, 100 for the 17.5k-utterance `esd_all` VAD sweep).
- `common.read_esd_speaker_txt`: BOM-based encoding detection. ESD's per-speaker
  transcript files mix utf-8 / utf-16 / latin-1, and Python < 3.13's incremental
  UTF-16 decoder raises a bare `UnicodeError` on BOM-less input, which the
  `except UnicodeDecodeError` chain copied from articulatory-tts does not catch
  (filed upstream as nzxyin/articulatory-tts#30).
- `scripts/run_synth.sh` / `run_score.sh`: `${1:-}` so the usage message is
  reachable under `set -u`. ESD scoring resubmitted with `--exclude=babel-l9-16`
  (onnxruntime fails to import on that node, which torchmetrics' DNSMOS reports as
  "not installed"; the same venv works everywhere else).
- Added `eval/inspect_results.py` (stdlib-only headline metrics, pred/GT duration
  ratio, worst-WER utterances) and `eval/summarize_results.py` (markdown table).
- Large `$HOME` caches moved to `/data` and symlinked back: `~/.local/share/tts`
  (845 MB, Coqui). `~/.cache/utmosv2` and `~/.cache/huggingface/hub` were already
  symlinked to `/data` (18:10 on 2026-09-07). The uv-managed interpreters under
  `~/.local/share/uv` (954 MB) stay in `$HOME` on purpose: `/data` is not mounted
  on the login node, so moving them would break `python3.1x`/`uv` there.

## 2026-09-07

- Cloned upstream (HEAD 3907152, "Create LICENSE"). Only upstream-code change:
  `models/tts/EmoSpherepp/model.py` imports `monotonic_align` lazily (training-only
  Cython MAS dependency; inference never touches it).
- Docker-free environment: `pyproject.toml` + `uv.lock` (Python 3.10.19, torch
  2.14.0+cu130, diffusers 0.35.2, transformers 4.57.6, funasr 1.4.12, speechbrain
  1.1.1, g2p_en, nltk 3.8.1, conformer, pyloudnorm). Venv on
  `/data/user_data/xoy/venvs/emospherepp`, `.venv` symlinked to it;
  `scripts/env.sh` overrides `UV_PROJECT_ENVIRONMENT` (the user's `~/.zshrc` pins
  uv to the articulatory-tts venv) and routes every cache to `/data`.
- `scripts/download_ckpt.sh`: resolves the README's Naver Works Drive link through
  `api.drive.worksmobile.com` (`/v1/shared-links/<key>` → `rootFileId`,
  `/files/<id>` → `downloadUrl`); `model_ckpt_steps_1100000.ckpt`, 338,738,261 B.
- Evaluation pipeline under `eval/` (see CLAUDE.md for the design):
  `build_phone_set.py`, `prepare_manifests.py`, `extract_ref_features.py`,
  `synthesize.py`, `score_wav_pairs.py`; SLURM stages `scripts/run_features.sh`,
  `run_synth.sh`, `run_score.sh`, chained by `scripts/run_all.sh`
  (`preempt`/`preempt_qos`, `--requeue`, restart-safe). Bootstrap + 3-utterance
  smoke test: `scripts/setup_env.sh` (job 10350507, preempted 4×, completed 18:10).
- Full chain launched 18:51 (features 10351016 → synth/score per test set, plus
  vocoder-resynthesis anchors for LJSpeech and ESD). Features (18 min), all four
  syntheses, and LJSpeech/LibriTTS scoring completed the same evening; the ESD
  scoring jobs failed on babel-l9-16 (see 2026-09-08).
