"""Extract EmoSphere++'s three reference-audio conditioning inputs for every
utterance in a manifest, caching one .pt per utterance under FEATURES_DIR:

  wavlm : 512-d x-vector, microsoft/wavlm-base-sv  (embedding_extract.WavLM_embeddings:
          librosa 16 kHz load, zero-pad to >= 1 s, Wav2Vec2FeatureExtractor, .embeddings)
  e2v   : 768-d emotion2vec+ BASE utterance embedding (embedding_extract.Emotion2Vec_embeddings
          used iic/emotion2vec_plus_base via modelscope; here funasr AutoModel(hub="hf") loads
          the identical weights). NOTE: base, not large -- the model's emo_mlp/emo_proj take 768.
  vad   : 3-d [arousal, dominance, valence] from audeering/wav2vec2-large-robust-12-ft-
          emotion-msp-dim, exactly as data_gen/tts/base_binarizer.py computes it (the wav
          fed in is librosa_wav2spec_bigvgan's peak-normalized (x0.95) float16 waveform).

`--esd_stats` then turns the `esd_all` VAD features into the per-emotion
distance quartiles (q1/q3) the binarizer's convert_tensor_to_polar_coordinates
normalizes intensity with. The released checkpoint's binarizer (base_binarizer.py)
computed them over the 8 SEEN speakers only; base_binarizer_unseen.py over the 2
unseen ones (0013, 0019) -- both subsets are written so synthesize.py can apply
the matching normalization per speaker.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EMO_MAP, FEATURES_DIR, UNSEEN_SPEAKERS, VAD_STATS, load_manifest  # noqa: E402

VAD_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
WAVLM_MODEL = "microsoft/wavlm-base-sv"
E2V_MODEL = "iic/emotion2vec_plus_base"

# Emotion centers hardcoded in base_binarizer.process_data (model output order:
# arousal, dominance, valence). Neutral = mean of neutral VAD; the others are the
# "I2I" ratio-optimized centers (VAD_analysis_I2I.py).
CENTERS = {
    2: [0.4135, 0.5169, 0.3620],
    0: [0.37068613, 0.4814421, 0.37709341],
    1: [0.36792182, 0.48303542, 0.34562907],
    3: [0.48519272, 0.57727589, 0.39524114],
    4: [0.36877737, 0.48431942, 0.36644742],
}


# ---- copied verbatim from data_gen/tts/base_binarizer.py (audeering model head) ----
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def _emotion_model_cls():
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            hidden_states_all = outputs[0]
            hidden_states = torch.mean(hidden_states_all, dim=1)
            logits = self.classifier(hidden_states)
            return hidden_states_all, hidden_states, logits

    return EmotionModel


def load_wav16(path):
    import librosa
    wav, _ = librosa.load(path, sr=16000)
    return wav.astype(np.float32)


class Extractors:
    def __init__(self, device, want):
        from transformers import Wav2Vec2FeatureExtractor
        self.device = device
        self.want = want
        if "vad" in want:
            self.vad_fe = Wav2Vec2FeatureExtractor.from_pretrained(VAD_MODEL)
            self.vad_model = _emotion_model_cls().from_pretrained(VAD_MODEL).to(device).eval()
        if "wavlm" in want:
            from transformers import WavLMForXVector
            self.wavlm_fe = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL)
            self.wavlm = WavLMForXVector.from_pretrained(WAVLM_MODEL).to(device).eval()
        if "e2v" in want:
            from funasr import AutoModel
            self.e2v = AutoModel(model=E2V_MODEL, hub="hf", device=str(device), disable_update=True)

    @torch.no_grad()
    def vad(self, wav):
        import librosa
        # base_binarizer: item['wav'] is librosa_wav2spec_bigvgan's output
        # (librosa.util.normalize(wav) * 0.95) stored as float16.
        w = (librosa.util.normalize(wav) * 0.95).astype(np.float16).astype(np.float32)
        x = self.vad_fe(w, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        _, _, logits = self.vad_model(x)
        return logits[0].float().cpu()

    @torch.no_grad()
    def wavlm_emb(self, wav):
        audio = wav
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)), "constant", constant_values=0)
        iv = self.wavlm_fe(audio, return_tensors="pt", sampling_rate=16000).input_values.to(self.device)
        return self.wavlm(iv).embeddings[0].float().cpu()

    @torch.no_grad()
    def e2v_emb(self, wav):
        res = self.e2v.generate([wav], fs=16000, granularity="utterance", extract_embedding=True)
        return torch.from_numpy(np.asarray(res[0]["feats"], dtype=np.float32).reshape(-1))


def extract(args):
    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    want = set(args.features.split(","))
    os.makedirs(FEATURES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    todo = []
    for r in rows:
        p = os.path.join(FEATURES_DIR, f"{r['uid']}.pt")
        have = set(torch.load(p, map_location="cpu").keys()) if os.path.exists(p) else set()
        if not want <= have:
            todo.append((r, p, have))
    print(f"{args.manifest}: {len(rows)} utterances, {len(todo)} need {sorted(want)} (device={device})")
    if not todo:
        return
    ex = Extractors(device, want)
    from tqdm import tqdm
    n_err = 0
    for r, p, have in tqdm(todo, desc=f"features[{args.manifest}]"):
        try:
            wav = load_wav16(r["wav"])
            feats = torch.load(p, map_location="cpu") if have else {}
            if "vad" in want and "vad" not in feats:
                feats["vad"] = ex.vad(wav)
            if "wavlm" in want and "wavlm" not in feats:
                feats["wavlm"] = ex.wavlm_emb(wav)
            if "e2v" in want and "e2v" not in feats:
                feats["e2v"] = ex.e2v_emb(wav)
            feats["emotion"] = r["emotion"]
            torch.save(feats, p + ".tmp")
            os.replace(p + ".tmp", p)
        except Exception as e:  # noqa: BLE001
            n_err += 1
            tqdm.write(f"ERROR {r['uid']}: {type(e).__name__}: {e}")
    print(f"done; {n_err} errors")
    if n_err > args.max_errors:
        sys.exit(1)


def esd_stats(args):
    """q1/q3 of ||VAD - center(emo)|| per emotion, per speaker subset (seen: 8
    speakers the checkpoint trained on; unseen: 0013/0019)."""
    rows = load_manifest("esd_all")
    subsets = {"seen": [], "unseen": []}
    for r in rows:
        subsets["unseen" if r["spk"] in UNSEEN_SPEAKERS else "seen"].append(r)
    out = {"centers": {str(k): v for k, v in CENTERS.items()}, "subsets": {}}
    for name, rs in subsets.items():
        dists = {e: [] for e in range(5)}
        n_missing = 0
        for r in rs:
            p = os.path.join(FEATURES_DIR, f"{r['uid']}.pt")
            if not os.path.exists(p):
                n_missing += 1
                continue
            vad = torch.load(p, map_location="cpu")["vad"].numpy()
            emo_id = EMO_MAP[r["emotion"]]
            # torch.nn.functional.pairwise_distance(p=2) adds eps=1e-6 to the
            # difference before the norm -- replicate exactly.
            dists[emo_id].append(float(np.linalg.norm(vad - np.array(CENTERS[emo_id]) + 1e-6)))
        out["subsets"][name] = {
            str(e): {"q1": float(np.percentile(d, 25)), "q3": float(np.percentile(d, 75)), "n": len(d)}
            for e, d in dists.items() if d
        }
        print(f"{name}: {len(rs)} utts ({n_missing} missing features) -> {json.dumps(out['subsets'][name])}")
    with open(VAD_STATS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {VAD_STATS}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="manifest name under MANIFEST_DIR (ljspeech|libritts|libritts_test_other|esd|esd_all)")
    ap.add_argument("--features", default="wavlm,e2v,vad", help="comma list of wavlm,e2v,vad")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_errors", type=int, default=0, help="exit non-zero if more utterances than this fail")
    ap.add_argument("--esd_stats", action="store_true", help="compute per-emotion VAD quartiles from esd_all features")
    args = ap.parse_args()
    if args.manifest:
        extract(args)
    if args.esd_stats:
        esd_stats(args)


if __name__ == "__main__":
    main()
