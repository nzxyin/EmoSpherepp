"""Per-utterance emotion2vec+ large cosine (pred vs GT) for a wav_pairs dir -- the
per-pair values that articulatory-tts/score_side_metric.py --metric emotion
aggregates away (it only writes {mean, ci95, n}). Same model (iic/emotion2vec_plus_large,
funasr, hub=hf), same extract_embedding=True utterance-level feats, same 0.1 s
minimum-duration guard, so the mean over all pairs reproduces that script's
emotion_cosine. Run under the eval-emotion venv. Resumable (rewrites --out every
100 pairs; existing uids skipped).
"""
import argparse
import glob
import json
import os

os.environ.setdefault("HF_HOME", "/data/hf_cache")

import numpy as np
import soundfile as sf

MIN_PAIR_DURATION_SEC = 0.1  # score_side_metric.py


def _too_short(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate < MIN_PAIR_DURATION_SEC


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_pairs(d):
    pairs = []
    for pred in sorted(glob.glob(os.path.join(d, "*_pred.wav"))):
        uid = os.path.basename(pred)[: -len("_pred.wav")]
        gt = os.path.join(d, f"{uid}_gt.wav")
        if os.path.exists(gt):
            pairs.append((uid, pred, gt))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav_pairs_dir", required=True)
    ap.add_argument("--out", required=True, help="JSON {uid: cosine}; uids skipped for length are recorded as null")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from funasr import AutoModel
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = find_pairs(args.wav_pairs_dir)
    if args.limit:
        pairs = pairs[: args.limit]
    done = json.load(open(args.out)) if os.path.exists(args.out) else {}
    todo = [p for p in pairs if p[0] not in done]
    print(f"{len(pairs)} pairs in {args.wav_pairs_dir}; {len(done)} already done; scoring {len(todo)}")
    if not todo:
        return
    model = AutoModel(model="iic/emotion2vec_plus_large", hub="hf", device=str(device))

    def embed(path):
        res = model.generate(path, granularity="utterance", extract_embedding=True)
        return np.asarray(res[0]["feats"]).reshape(-1)

    def flush():
        with open(args.out + ".tmp", "w") as f:
            json.dump(done, f, indent=1)
        os.replace(args.out + ".tmp", args.out)

    for i, (uid, pred, gt) in enumerate(tqdm(todo, desc="emotion_cosine per utt")):
        if _too_short(pred) or _too_short(gt):
            tqdm.write(f"WARNING: skipping {uid} -- too short for emotion2vec (min={MIN_PAIR_DURATION_SEC}s)")
            done[uid] = None
        else:
            done[uid] = cosine(embed(pred), embed(gt))
        if (i + 1) % 100 == 0:
            flush()
    flush()
    vals = [v for v in done.values() if v is not None]
    print(f"wrote {args.out}: n={len(vals)} mean={np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
