"""Standalone EmoSphere++ inference for an evaluation manifest -> {uid}_pred.wav /
{uid}_gt.wav pairs (16 kHz), the layout articulatory-tts's score_side_metric.py
consumes and the one score_wav_pairs.py scores.

Why not tasks/run.py --infer: that path needs the never-released NATSpeech
binarized ESD data plus hard-coded /workspace/... feature dirs
(tasks/tts/dataset_utils.py). This script reproduces the same forward pass
(tasks/tts/EmoSpherepp.py forward(infer=True): n_timesteps=100, guidance_scale=0,
temperature 1.0, length_scale 1.0) from raw text + a reference wav:

  text --(data_gen/tts/txt_processors/en.py + phone_set.json + intersperse blank)--> tokens
  reference wav --(extract_ref_features.py)--> WavLM x-vector, emotion2vec+ base, VAD
  VAD --(base_binarizer.convert_tensor_to_polar_coordinates)--> [intensity, elevation, azimuth]
  model.forward -> 80-bin log-mel (16 kHz, hop 256) -> vocoder -> peak-normalized 16-bit wav
  (the original BigVGAN wrapper + utils/audio/io.save_wav also end in peak normalization)

Reference = the target utterance's own ground-truth recording ("paired"
reference), matching articulatory-tts's eval, which conditions on the target
utterance's own speaker embedding. --ref_mode pair (a different utterance of the
same speaker, EmoSphere++'s "up_" setting) is available for ESD via test_pair_wav.txt.

Vocoder: speechbrain/tts-hifigan-libritts-16kHz -- the paper's BigVGAN-16k
checkpoint was never released (README), and this is the public 16 kHz vocoder
whose mel parameterization (80 bands, hop 256, n_fft/win 1024, fmin 0, fmax 8000,
natural-log magnitude, slaney filterbank) matches the one EmoSphere++ was trained on
(egs/egs_bases/tts/dataset_params.yaml + utils/audio/librosa_wav2spec_bigvgan).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))

from common import (CKPT_PATH, EMO_MAP, FEATURES_DIR, NEUTRAL_ID, PHONE_SET, UNSEEN_SPEAKERS,  # noqa: E402
                    VAD_STATS, WAV_PAIRS_ROOT, load_manifest)
from extract_ref_features import CENTERS  # noqa: E402

PREPROCESS_ARGS = {"txt_processor": "en", "with_phsep": True, "add_eos_bos": True}
VOCODER_SOURCE = "speechbrain/tts-hifigan-libritts-16kHz"
SR = 16000


def spherical_vector(vad, emo_id, q1, q3):
    """base_binarizer.convert_tensor_to_polar_coordinates, numpy, one utterance.
    Returns (intensity r_norm, elevation theta, azimuth phi)."""
    v = np.asarray(vad, dtype=np.float64) - np.asarray(CENTERS[emo_id], dtype=np.float64)
    r = float(np.sqrt(np.sum(v ** 2)))
    iqr = (q3 - q1) * 1.5
    max_emo, min_emo = q3 + iqr, q1 - iqr
    r_clamp = min(max(r, min_emo), max_emo)
    r_norm = (r_clamp - min_emo) / (max_emo - min_emo)
    theta = float(np.arccos(np.clip(v[2] / r, -1.0, 1.0))) if r > 0 else 0.0
    phi = float(np.arctan2(v[1], v[0]))
    if r_norm == 0:
        theta = phi = 0.0
    if emo_id == NEUTRAL_ID:
        r_norm = theta = phi = 0.0
    return float(r_norm), theta, phi


# ---- mel for --resynth (copy of utils/audio.mel_spectrogram + librosa_wav2spec_bigvgan,
# duplicated here because utils/audio/__init__.py imports pyworld/webrtcvad at module level) ----
_mel_basis, _hann = {}, {}


def gt_mel(wav16, n_fft=1024, num_mels=80, hop=256, win=1024, fmin=0, fmax=8000):
    import librosa
    from librosa.filters import mel as librosa_mel_fn
    y = torch.FloatTensor(librosa.util.normalize(wav16) * 0.95).unsqueeze(0)
    key = f"{fmax}_{y.device}"
    if key not in _mel_basis:
        _mel_basis[key] = torch.from_numpy(librosa_mel_fn(sr=SR, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)).float()
        _hann[str(y.device)] = torch.hann_window(win)
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop) / 2), int((n_fft - hop) / 2)), mode="reflect").squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop, win_length=win, window=_hann[str(y.device)], center=False,
                      pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    linear = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    mel = torch.log(torch.clamp(torch.matmul(_mel_basis[key], linear), min=1e-5))
    return mel.squeeze(0).T  # [T, 80]


def gt_mel_native(wav16, n_fft=1024, num_mels=80, hop=256, win=1024, fmin=0, fmax=8000):
    """The VOCODER's own training convention (speechbrain recipes/LibriTTS/vocoder/hifigan:
    torchaudio MelSpectrogram, center=True / n_fft//2 reflect padding, power=1, slaney,
    log(clamp(x, 1e-5))) -- differs from gt_mel() only by the STFT centering (half-hop
    frame offset, one extra frame). Used by --mel_center speechbrain to measure how much
    that framing difference costs the anchor."""
    import librosa
    import torchaudio
    y = torch.FloatTensor(librosa.util.normalize(wav16) * 0.95).unsqueeze(0)
    tf = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=n_fft, win_length=win, hop_length=hop,
                                              f_min=fmin, f_max=fmax, n_mels=num_mels, power=1.0,
                                              normalized=False, norm="slaney", mel_scale="slaney")
    mel = torch.log(torch.clamp(tf(y), min=1e-5))
    return mel.squeeze(0).T  # [T, 80]


def load_model(device):
    from utils.commons.hparams import set_hparams
    from utils.commons.ckpt_utils import load_ckpt
    from utils.text.text_encoder import build_token_encoder
    from models.tts.EmoSpherepp import EmoSpherepp

    hp = set_hparams(config=os.path.join(REPO, "egs/egs_bases/tts/emospherepp.yaml"), print_hparams=False, global_hparams=False)
    encoder = build_token_encoder(PHONE_SET)
    model = EmoSpherepp(len(encoder) + 1, hp, out_dims=hp["audio_num_mel_bins"])
    load_ckpt(model, CKPT_PATH, model_name="model", strict=True)
    model.to(device).eval()
    global MAX_INPUT_TOKENS
    MAX_INPUT_TOKENS = int(hp["max_input_tokens"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"EmoSphere++ loaded: {n_params / 1e6:.1f}M params, vocab {len(encoder)}+1 blank, ckpt {CKPT_PATH}")
    return model, encoder, hp


def load_vocoder(device):
    from speechbrain.inference.vocoders import HIFIGAN
    savedir = os.path.join(os.environ.get("HF_HOME", "/tmp"), "tts-hifigan-libritts-16kHz")
    voc = HIFIGAN.from_hparams(source=VOCODER_SOURCE, savedir=savedir, run_opts={"device": str(device)})
    print(f"vocoder loaded: {VOCODER_SOURCE}")
    return voc


MAX_INPUT_TOKENS = 1550  # egs/egs_bases/tts/base.yaml; overwritten from hparams in load_model()


def text_to_tokens(text, encoder):
    from data_gen.tts.txt_processors.en import TxtProcessor
    from utils.text import intersperse
    txt_struct, _ = TxtProcessor.process(text, PREPROCESS_ARGS)
    ph = [p for _w, phs in txt_struct for p in phs]
    n_oov = sum(1 for p in ph if p not in encoder.token_to_id)
    tokens = encoder.encode(" ".join(ph))
    if len(tokens) > MAX_INPUT_TOKENS:  # BaseSpeechDataset.__getitem__ truncates ph_token BEFORE intersperse
        print(f"WARNING: truncating {len(tokens)} phone tokens to max_input_tokens={MAX_INPUT_TOKENS}")
        tokens = tokens[:MAX_INPUT_TOKENS]
    tokens = intersperse(tokens, len(encoder))
    return tokens, ph, n_oov


def seed_for(uid, base=1234):
    return base + int(hashlib.md5(uid.encode()).hexdigest()[:8], 16) % (2 ** 31 - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["ljspeech", "libritts", "libritts_test_other", "esd"])
    ap.add_argument("--out_dir", default=None, help="wav-pairs dir (default WAV_PAIRS_ROOT/<dataset>[_resynth])")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n_timesteps", type=int, default=100, help="Euler steps (tasks/tts/EmoSpherepp.py uses 100)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--length_scale", type=float, default=1.0)
    ap.add_argument("--ref_mode", choices=["self", "pair"], default="self",
                    help="self: reference = the target utterance's own GT (default). pair: ESD test_pair_wav.txt partner")
    ap.add_argument("--resynth", action="store_true", help="vocoder ceiling: GT wav -> mel -> vocoder (no TTS)")
    ap.add_argument("--mel_center", choices=["hifigan", "speechbrain"], default="hifigan",
                    help="--resynth only: mel framing. hifigan = EmoSphere++'s own convention (center=False, "
                         "(n_fft-hop)/2 reflect pad; what the model emits); speechbrain = the vocoder's native "
                         "torchaudio center=True convention (out_dir suffix _resynth_native)")
    ap.add_argument("--no_gt", action="store_true", help="don't (re)write {uid}_gt.wav")
    args = ap.parse_args()

    rows = load_manifest(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    suffix = ("_resynth_native" if args.mel_center == "speechbrain" else "_resynth") if args.resynth else ""
    out_dir = args.out_dir or os.path.join(WAV_PAIRS_ROOT, args.dataset + suffix)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{args.dataset}: {len(rows)} utterances -> {out_dir} (device={device}, resynth={args.resynth}, ref={args.ref_mode})")

    pair_map = {}
    if args.ref_mode == "pair":
        with open(os.path.join(REPO, "test_pair_wav.txt")) as f:
            for line in f:
                a, b = [x.strip() for x in line.strip().split(",")]
                pair_map[a] = b

    vocoder = load_vocoder(device)
    model = encoder = None
    stats = json.load(open(VAD_STATS)) if os.path.exists(VAD_STATS) else None
    if not args.resynth:
        model, encoder, _hp = load_model(device)

    import librosa
    from tqdm import tqdm
    log_path = os.path.join(out_dir, "synth_log.jsonl")
    done = set()
    if os.path.exists(log_path):
        with open(log_path) as f:
            for l in f:
                if not l.strip():
                    continue
                try:
                    done.add(json.loads(l)["uid"])
                except (json.JSONDecodeError, KeyError):
                    print("WARNING: dropping truncated synth_log line (job was likely preempted mid-write)")
    n_skip = n_err = 0
    with open(log_path, "a") as log:
        for r in tqdm(rows, desc=f"synth[{args.dataset}]"):
            uid = r["uid"]
            pred_path = os.path.join(out_dir, f"{uid}_pred.wav")
            gt_path = os.path.join(out_dir, f"{uid}_gt.wav")
            if uid in done and os.path.exists(pred_path) and (args.no_gt or os.path.exists(gt_path)):
                n_skip += 1
                continue
            t0 = time.time()
            try:
                gt, gt_sr = sf.read(r["wav"], dtype="float32")
                if gt.ndim > 1:
                    gt = gt.mean(axis=1)
                gt16 = gt if gt_sr == SR else librosa.resample(gt, orig_sr=gt_sr, target_sr=SR)
                if not args.no_gt:
                    sf.write(gt_path, gt16.astype(np.float32), SR)

                rec = {"uid": uid, "emotion": r["emotion"], "spk": r["spk"]}
                if args.resynth:
                    mel = (gt_mel_native(gt16) if args.mel_center == "speechbrain" else gt_mel(gt16)).to(device)  # [T,80]
                else:
                    ref_uid = pair_map.get(uid, uid) if args.ref_mode == "pair" else uid
                    feats = torch.load(os.path.join(FEATURES_DIR, f"{ref_uid}.pt"), map_location="cpu")
                    emo_id = EMO_MAP[r["emotion"]]
                    if emo_id == NEUTRAL_ID:
                        inten, theta, phi = 0.0, 0.0, 0.0
                    else:
                        subset = "unseen" if r["spk"] in UNSEEN_SPEAKERS else "seen"
                        q = stats["subsets"][subset][str(emo_id)]
                        inten, theta, phi = spherical_vector(feats["vad"].numpy(), emo_id, q["q1"], q["q3"])
                    tokens, ph, n_oov = text_to_tokens(r["text"], encoder)
                    x = torch.LongTensor(tokens)[None].to(device)
                    x_len = torch.LongTensor([len(tokens)]).to(device)
                    torch.manual_seed(seed_for(uid))
                    with torch.no_grad():
                        out = model(
                            x, x_len, n_timesteps=args.n_timesteps, temperature=args.temperature,
                            spk=feats["wavlm"][None].to(device), emo=feats["e2v"][None].to(device),
                            inten_vector=torch.tensor([[[inten]]], dtype=torch.float32, device=device),
                            style_vector=torch.tensor([[[theta, phi]]], dtype=torch.float32, device=device),
                            length_scale=args.length_scale, guidance_scale=0.0,
                        )
                    mel = out["mel_out"][0]  # [T,80]
                    rec.update({"ref": ref_uid, "n_tokens": len(tokens), "n_ph": len(ph), "n_oov": n_oov,
                                "inten": inten, "theta": theta, "phi": phi,
                                "vad": [float(v) for v in feats["vad"].numpy()]})
                with torch.no_grad():
                    wav = vocoder.decode_batch(mel.T.unsqueeze(0).to(device)).squeeze().float().cpu().numpy()
                peak = float(np.abs(wav).max()) if wav.size else 0.0
                if peak > 0:
                    wav = wav / peak
                sf.write(pred_path, (wav * 32767).astype(np.int16), SR, subtype="PCM_16")
                rec.update({"n_frames": int(mel.shape[0]), "pred_sec": len(wav) / SR, "gt_sec": len(gt16) / SR,
                            "elapsed": time.time() - t0})
                log.write(json.dumps(rec) + "\n")
                log.flush()
            except Exception as e:  # noqa: BLE001
                n_err += 1
                tqdm.write(f"ERROR {uid}: {type(e).__name__}: {e}")
    n_pred = len([r for r in rows if os.path.exists(os.path.join(out_dir, f"{r['uid']}_pred.wav"))])
    print(f"done: {len(rows) - n_skip - n_err} synthesized, {n_skip} skipped (existing), {n_err} errors -> {out_dir}; "
          f"{n_pred}/{len(rows)} manifest utterances have a _pred.wav")
    if n_err or n_pred != len(rows):
        sys.exit(1)  # non-zero so a SLURM afterok scoring dependency does not run on an incomplete set


if __name__ == "__main__":
    main()
