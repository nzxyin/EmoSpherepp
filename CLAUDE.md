# EmoSpherepp (nzxyin/EmoSpherepp — fork of Choddeok/EmoSpherepp for baseline evaluation)

Fork of the official EmoSphere++ code (Cho et al., IEEE TAFFC 2025; `upstream` remote): emotion-controllable
zero-shot TTS, NATSpeech-based training framework, Matcha-TTS-style flow-matching
decoder, 16 kHz / 80-bin mel. Upstream is unchanged since 2025-04 apart from README
and LICENSE (checked 2026-09-07; no releases, 3 open issues all asking how to run
inference). This clone adds a Docker-free `uv` environment and an evaluation
pipeline under `eval/` + `scripts/`; the upstream code is untouched except one
lazy import (`models/tts/EmoSpherepp/model.py`: `monotonic_align` is training-only).

Cluster conventions (partitions, `/data`, login-node limits) live in the user-level
`~/.claude/CLAUDE.md`, not here.

## Goal of this clone (2026-09-07)

Evaluate the released EmoSphere++ checkpoint as a baseline on the SAME held-out
test sets and metric stack as `/home/xoy/articulatory-tts` (`eval_full_testset.py`):
LJSpeech test (n=150), LibriTTS-R test-clean (~4830), test-other (~5106), ESD
English test (1500); metrics Whisper-large-v3 corpus WER, UTMOSv2, DNSMOS,
ECAPA speaker cosine, emotion2vec+ large emotion cosine.

## Environment (no Docker)

- `pyproject.toml` + `uv.lock` (Python 3.10, torch 2.14+cu130, diffusers<0.36,
  conformer, funasr, speechbrain, g2p_en, nltk<3.9). The venv lives on `/data`
  (`/data/user_data/xoy/venvs/emospherepp`), `.venv` is a symlink to it.
- `scripts/env.sh` — every job sources it. It MUST override
  `UV_PROJECT_ENVIRONMENT` (the user's `~/.zshrc` points uv at the
  articulatory-tts venv, and sbatch inherits that), and routes HF/uv/nltk caches
  to `/data`.
- Bootstrap + smoke test: `sbatch scripts/setup_env.sh` (preempt, 1 GPU). Builds
  the venv, downloads the checkpoint and models, rebuilds `phone_set.json`,
  writes manifests, synthesizes 3 utterances, scores them.
- Checkpoint: `scripts/download_ckpt.sh` resolves the README's works.do link via
  `api.drive.worksmobile.com` (`model_ckpt_steps_1100000.ckpt`, 338,738,261 B,
  global step 1.1M, trained on the 8 seen ESD speakers; 0013/0019 held out).

## How inference works here (`eval/`)

The repo's own `tasks/run.py --infer` needs the never-released NATSpeech
binarized ESD data plus hard-coded `/workspace/...` feature dirs, so inference is
reproduced standalone in `eval/synthesize.py` with the identical forward pass
(`n_timesteps=100`, `guidance_scale=0`, temperature 1.0):

1. `eval/build_phone_set.py` — the checkpoint's `phone_set.json` was never
   shipped; it is `sorted(set(phonemes))` of `esd_text_emo.txt` through the repo's
   g2p_en front-end (`data_gen/tts/txt_processors/en.py`, `|` word separators,
   `<BOS>/<EOS>`). Vocab = 3 reserved + phones (minus `<EOS>`); model embedding has
   vocab+1 rows (intersperse blank). Verified against `encoder.emb.weight`.
2. `eval/prepare_manifests.py` — uid/text/wav/emotion per test set from the
   articulatory-tts split files (paths in `eval/common.py`). LJSpeech/LibriTTS
   have no emotion labels → "Neutral" (emo_id 2 → zero spherical vector).
3. `eval/extract_ref_features.py` — per reference wav: WavLM x-vector
   (`microsoft/wavlm-base-sv`, 512-d), emotion2vec+ **base** (768-d; the model's
   `emo_mlp` takes 768, so not "large" despite the README), audeering VAD
   (arousal/dominance/valence). `--esd_stats` recomputes the per-emotion
   distance quartiles (`convert_tensor_to_polar_coordinates`) over all 17.5k ESD
   utterances, separately for the seen and unseen speaker subsets, matching
   `base_binarizer.py` / `base_binarizer_unseen.py`.
4. `eval/synthesize.py` — text → tokens, features → spherical vector →
   `model.forward` → mel → vocoder → peak-normalized 16-bit wav; writes
   `{uid}_pred.wav` + `{uid}_gt.wav` (16 kHz) in `wav_pairs/<dataset>/`.
   Reference = the target utterance's own GT ("paired"), the same condition as
   articulatory-tts (which uses the target's own speaker embedding);
   `--ref_mode pair` uses `test_pair_wav.txt` partners (ESD only, EmoSphere++'s
   "up_" setting). `--resynth` = GT mel → vocoder (vocoder-ceiling anchor).
5. Vocoder: the paper's BigVGAN-16k was never released. We use
   `speechbrain/tts-hifigan-libritts-16kHz`, the public 16 kHz vocoder with the
   same mel parameterization (80 bands, hop 256, n_fft/win 1024, fmin 0/fmax 8000,
   natural-log magnitude, slaney). Absolute quality numbers therefore include a
   vocoder mismatch penalty vs. the paper — report the `--resynth` anchor alongside.
6. `eval/score_wav_pairs.py` — the scoring half of articulatory-tts's
   `eval_full_testset.score()` over wav pairs (same models, corpus-level WER,
   same JSON schema); runs under `/data/user_data/xoy/venvs/eval-articulatory-tts`.
   `emotion_cosine` reuses articulatory-tts's `score_side_metric.py` +
   `merge_eval_results.py` unchanged (eval-emotion venv).

Known, deliberate deviations from the original pipeline (reviewed 2026-09-08):
the per-emotion VAD quartiles include the handful of ESD utterances the binarizer
would have dropped for empty F0; `spherical_vector` clips the `acos` argument to
[-1, 1] where the original could produce NaN; predicted wavs are peak-normalized
16-bit (the original's `save_wav` does the same) while GT pairs are raw float32
(as in `eval_full_testset.py`). The vocoder was trained with torchaudio's
center=True framing while EmoSphere++ emits center=False (half-hop offset) mels;
`synthesize.py --resynth --mel_center speechbrain` produces a vocoder-native
anchor so that effect can be measured (see Status).

Jobs: `scripts/run_features.sh` → `scripts/run_synth.sh <dataset>` →
`scripts/run_score.sh <dataset> [tag]`; all on `preempt`/`preempt_qos`,
`--requeue`, restart-safe (skip existing outputs). Outputs under
`/data/user_data/xoy/EmoSpherepp_eval/` (ckpt, features, manifests, wav_pairs,
results). `eval/summarize_results.py` tabulates `results/eval_*.json`.

## Status / results (2026-09-08, complete)

Full chain ran 2026-09-07 evening (`scripts/run_all.sh`; features 18 min, synthesis
1–74 min per set, scoring ~2 s/utt). Reference = each utterance's own GT
("paired"); LJSpeech/LibriTTS conditioned as Neutral (zero spherical vector);
ESD uses its labels + VAD-derived intensity/style with the seen/unseen quartiles.
ESD scoring completed 2026-09-08 01:20 after a rerun off `babel-l9-16` (onnxruntime
import failure there). All numbers: `/data/user_data/xoy/EmoSpherepp_eval/results/`
(`eval_<tag>.json`, per-utterance `eval_<tag>_per_utt.json`).

| test set | n | WER | UTMOSv2 | DNSMOS_ovr | speaker_cos | emotion_cos | pred/GT dur |
|---|---|---|---|---|---|---|---|
| LJSpeech test | 150 | 10.60% | 2.926 ±0.042 | 3.159 ±0.020 | 0.173 ±0.011 | 0.880 ±0.018 | 1.20 |
| LibriTTS-R test-clean | 4830 | 14.54% | 2.912 ±0.011 | 3.141 ±0.005 | 0.270 ±0.004 | 0.858 ±0.004 | 1.36 |
| LibriTTS-R test-other | 5106 | 16.55% | 2.863 ±0.011 | 3.130 ±0.005 | 0.225 ±0.003 | 0.837 ±0.004 | 1.44 |
| ESD test | 1500 | 18.96% | 2.683 ±0.023 | 3.076 ±0.011 | 0.567 ±0.012 | 0.933 ±0.005 | 1.10 |
| *anchor:* ESD GT mel → HiFi-GAN | 1500 | 15.67% | 2.996 ±0.025 | 3.093 ±0.012 | 0.950 ±0.001 | 0.974 ±0.003 | 1.05 |
| *anchor:* LJSpeech GT mel → HiFi-GAN | 150 | 7.04% | 3.398 ±0.044 | 3.081 ±0.047 | 0.960 ±0.002 | 0.984 ±0.003 | 1.02 |
| *anchor:* same, vocoder-native (center=True) mel | 150 | 7.16% | 3.440 ±0.040 | 3.092 ±0.044 | 0.961 ±0.002 | 0.985 ±0.003 | 1.03 |

For comparison, articulatory-tts's own anchors on the identical sets (its
`sparc_resynth_baselines/`): raw GT WER 7.0 / 10.6 / 13.5 / 15.1% (LJSpeech /
clean / other / ESD); its best models (CLAUDE.md, MSE/soft-DTW + large_dim):
LJSpeech WER 6.55%, UTMOSv2 3.15–3.24, DNSMOS_ovr 3.30–3.34, speaker_cos 0.44–0.49;
test-clean 8.4–8.5%, 3.02–3.12, 3.22–3.25, 0.69; test-other 10.7–11.0%,
2.97–3.09, 3.17–3.19, 0.61; ESD 12.8–13.9%, 2.47–2.53, 2.90–2.92, 0.40–0.41,
emotion_cos 0.57–0.60.

ESD breakdown (`eval_esd.json.partial.jsonl`): seen speakers (n=1200) WER 18.7%,
UTMOSv2 2.70, speaker_cos **0.670**; unseen 0013/0019 (n=300) WER 20.1%, UTMOSv2
2.63, speaker_cos **0.154**. Per emotion (WER / UTMOSv2 / DNSMOS_ovr): Neutral 17.0
/ 2.93 / 3.11, Angry 17.2 / 2.79 / 3.19, Happy 18.5 / 2.67 / 3.08, Sad 18.8 / 2.62 /
3.02, Surprise 23.3 / 2.39 / 2.98.

Findings:
- **Speaker cloning works for seen speakers only**: ESD seen 0.67 vs unseen 0.15,
  and 0.17–0.27 on LJSpeech/LibriTTS — the WavLM x-vector conditioning, trained on
  8 ESD voices, does not generalize zero-shot. The vocoder is not the cause (GT
  resynthesis gives 0.95–0.96 everywhere).
- **Emotion**: ESD emotion_cos 0.933 (anchor 0.974) — the emotion2vec+ reference
  embedding and VAD-derived spherical vector transfer the reference's emotion well;
  articulatory-tts reaches 0.57–0.60 on the same set (its emotion conditioning is a
  label, not a reference embedding, so the comparison favours EmoSphere++ by design).
  Surprise is the weakest emotion on every metric (WER 23%, UTMOSv2 2.39).
- **Intelligibility**: EmoSphere++ WER is 1.5–2× articulatory-tts on the read-speech
  sets (10.6 vs 6.6%, 14.5 vs 8.5%, 16.6 vs 10.7%) — and 3.6 pp above the vocoder
  ceiling on LJSpeech (7.0%), so it is the acoustic model, not the vocoder.
- **Speaker similarity is the weak spot**: 0.17–0.27 vs 0.44–0.69 for
  articulatory-tts, while the same vocoder resynthesizes GT at 0.96 — an
  8-speaker-ESD-trained x-vector conditioning does not transfer to unseen
  LJSpeech/LibriTTS voices (emotion2vec/x-vector zero-shot generalization, not a
  pipeline artifact).
- **Quality**: UTMOSv2 2.86–2.93 and DNSMOS_ovr 3.13–3.16 sit ~0.2–0.3 below
  articulatory-tts on the read-speech sets; on ESD (in-domain for EmoSphere++)
  UTMOSv2 2.68 / DNSMOS 3.08 beat articulatory-tts's 2.47–2.53 / 2.90–2.92.
  Resynthesis UTMOSv2 3.40 (LJSpeech) / 3.00 (ESD) shows the HiFi-GAN vocoder
  itself costs ~0.1–0.55 UTMOSv2 vs raw GT (3.95 / 3.11).
- **ESD WER 19.0%** vs 15.1% raw-GT / 15.7% anchor floor and articulatory-tts's
  12.8–13.9%: on emotional speech the acoustic model adds ~3 pp over the floor,
  the same gap as on LJSpeech; articulatory-tts's TTS output is *easier* for
  Whisper than the emotional GT itself.
- **Vocoder framing mismatch is negligible**: feeding the vocoder EmoSphere++'s
  center=False mels vs. its native center=True mels changes the LJSpeech anchor by
  +0.04 UTMOSv2 / +0.01 DNSMOS / +0.1 pp WER — all inside the ±ci95. The vocoder
  substitution itself (HiFi-GAN vs. the unreleased BigVGAN-16k) remains the one
  unquantifiable gap vs. the paper.
- **Pace**: outputs are 20–45% longer than GT on read speech (ESD ~9%): the ESD-trained
  duration predictor speaks slowly on long audiobook sentences. Whisper truncates at
  30 s, which penalizes the longest LibriTTS items (per-utt macro WER 19.5/24.0%
  vs corpus 14.5/16.6%). Worst-WER cases are mostly punctuation/normalization
  (refs keep quotes/dashes, same lower-case-only scoring as articulatory-tts).
