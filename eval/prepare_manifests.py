"""Build per-dataset manifests (JSONL: uid, text, wav, emotion, spk) for the four
articulatory-tts test splits, plus `esd_all` (every ESD English utterance, used
only to compute the per-emotion VAD-distance quartiles the spherical emotion
vector is normalized with -- see extract_ref_features.py).

Emotion labels: ESD's own per-speaker txt (Angry/Happy/Neutral/Sad/Surprise).
LJSpeech / LibriTTS-R have no emotion labels -> "Neutral" (emo_id 2), which is
exactly the case where EmoSphere++ zeroes the spherical vector (intensity 0,
style (0,0)); the emotion2vec+ reference embedding still carries the
reference's actual style.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATASET_SPECS, ESD_SPEAKERS, MANIFEST_DIR, index_wavs,  # noqa: E402
                    load_reference_texts, load_split_ids, read_esd_speaker_txt)


def build(dataset):
    spec = DATASET_SPECS[dataset]
    ids = load_split_ids(dataset)
    wav_index = index_wavs(spec["raw_wav_dir"])
    texts = load_reference_texts(dataset, ids)
    emotions = {}
    if dataset == "esd":
        for spk in sorted({u.split("_")[0] for u in ids}):
            for stem, (_txt, emo) in read_esd_speaker_txt(spec["raw_wav_dir"], spk).items():
                emotions[stem] = emo
    rows, missing = [], {"wav": 0, "text": 0, "emotion": 0}
    for uid in ids:
        wav = wav_index.get(uid)
        txt = texts.get(uid)
        if wav is None:
            missing["wav"] += 1
            continue
        if txt is None:
            missing["text"] += 1
            continue
        if dataset == "esd":
            emo = emotions.get(uid)
            if emo is None:
                missing["emotion"] += 1
                continue
            spk = uid.split("_")[0]
        else:
            emo = "Neutral"
            spk = uid.split("_")[0] if dataset.startswith("libritts") else "LJ"
        rows.append({"uid": uid, "text": txt, "wav": wav, "emotion": emo, "spk": spk})
    print(f"{dataset}: {len(ids)} ids -> {len(rows)} manifest rows; missing={missing}")
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    with open(os.path.join(MANIFEST_DIR, f"{dataset}.stats.json"), "w") as f:
        json.dump({"n_ids": len(ids), "n_rows": len(rows), "missing": missing,
                   "split_file": os.path.basename(spec["split_path"])}, f, indent=1)
    return rows


def build_esd_all():
    spec = DATASET_SPECS["esd"]
    wav_index = index_wavs(spec["raw_wav_dir"])
    rows = []
    for spk in ESD_SPEAKERS:
        table = read_esd_speaker_txt(spec["raw_wav_dir"], spk)
        n = 0
        for stem, (txt, emo) in sorted(table.items()):
            wav = wav_index.get(stem)
            if wav is None or emo is None:
                continue
            rows.append({"uid": stem, "text": txt, "wav": wav, "emotion": emo, "spk": spk})
            n += 1
        print(f"  esd_all {spk}: {n} utterances")
    print(f"esd_all: {len(rows)} rows")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ljspeech", "libritts", "libritts_test_other", "esd", "esd_all"])
    args = ap.parse_args()
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    for ds in args.datasets:
        rows = build_esd_all() if ds == "esd_all" else build(ds)
        out = os.path.join(MANIFEST_DIR, f"{ds}.jsonl")
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
