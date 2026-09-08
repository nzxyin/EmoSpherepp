"""Shared paths/constants for the EmoSphere++ evaluation pipeline on Babel.

Everything large lives on /data (not mounted on the login node -- run all of
this via sbatch). Dataset roots and split files are the SAME ones
articulatory-tts's eval_full_testset.py uses (copied from its DATASET_SPECS,
2026-09-07), so the four test sets are identical utterance lists:
  ljspeech            -> /data/user_data/xoy/LJSpeech-1.1/preprocessed/test.json      (n=150)
  libritts            -> /data/user_data/xoy/LibriTTS_R/test-clean.json               (n~4830)
  libritts_test_other -> /data/user_data/xoy/LibriTTS_R/test-other.json               (n~5106)
  esd                 -> /data/user_data/xoy/esd_english_splits/test.tsv              (n=1500)
"""
import csv
import glob
import json
import os

EVAL_ROOT = "/data/user_data/xoy/EmoSpherepp_eval"
CKPT_PATH = f"{EVAL_ROOT}/ckpt/model_ckpt_steps_1100000.ckpt"
PHONE_SET = f"{EVAL_ROOT}/phone_set.json"
FEATURES_DIR = f"{EVAL_ROOT}/features"          # per-utterance reference features (.pt)
VAD_STATS = f"{EVAL_ROOT}/features/esd_vad_stats.json"
MANIFEST_DIR = f"{EVAL_ROOT}/manifests"
WAV_PAIRS_ROOT = f"{EVAL_ROOT}/wav_pairs"
RESULTS_DIR = f"{EVAL_ROOT}/results"

# ESD emotion-name -> emo_id, exactly as BasePreprocessor.build_emo_map builds
# it (sorted emotion folder names): Angry=0, Happy=1, Neutral=2, Sad=3, Surprise=4.
EMO_MAP = {"Angry": 0, "Happy": 1, "Neutral": 2, "Sad": 3, "Surprise": 4}
NEUTRAL_ID = 2
# ESD English speakers 0011..0020 -> spk_id 0..9 (sorted). The released
# checkpoint was trained with spk_id 2 and 8 (0013, 0019) held out as UNSEEN
# (base_binarizer.py `if i == 2 or i == 8: continue`; confirmed in GH issue #12).
ESD_SPEAKERS = [f"{i:04d}" for i in range(11, 21)]
UNSEEN_SPEAKERS = {"0013", "0019"}

DATASET_SPECS = {
    "ljspeech": {
        "split_path": "/data/user_data/xoy/LJSpeech-1.1/preprocessed/test.json",
        "raw_wav_dir": "/data/user_data/xoy/LJSpeech-1.1/wavs",
        "metadata_csv": "/data/user_data/xoy/LJSpeech-1.1/metadata.csv",
    },
    "libritts": {
        "split_path": "/data/user_data/xoy/LibriTTS_R/test-clean.json",
        "raw_wav_dir": "/data/user_data/xoy/LibriTTS_R/test-clean",
    },
    "libritts_test_other": {
        "split_path": "/data/user_data/xoy/LibriTTS_R/test-other.json",
        "raw_wav_dir": "/data/group_data/UTD-NAS/Databases/LibriTTS-R/LibriTTS_R/test-other",
    },
    "esd": {
        "split_path": "/data/user_data/xoy/esd_english_splits/test.tsv",
        "raw_wav_dir": "/data/group_data/UTD-NAS/Databases/ESD/ESD",
    },
}


def load_split_ids(dataset):
    spec = DATASET_SPECS[dataset]
    p = spec["split_path"]
    if p.endswith(".tsv"):
        with open(p, newline="") as f:
            return [row["stem"] for row in csv.DictReader(f, delimiter="\t")]
    return json.load(open(p))


def index_wavs(raw_wav_dir):
    """stem -> path, one os.walk (same trick as eval_full_testset.py's wav_index)."""
    idx = {}
    for root, _dirs, files in os.walk(raw_wav_dir):
        for fn in files:
            if fn.endswith(".wav"):
                idx.setdefault(fn[:-4], os.path.join(root, fn))
    return idx


def read_esd_speaker_txt(raw_wav_dir, spk):
    """ESD's per-speaker <spk>/<spk>.txt (tab-separated stem / transcript /
    emotion). Mixed encodings across speakers -- same utf-8-sig -> utf-16 ->
    latin-1 fallback chain as eval_full_testset.load_reference_texts."""
    txt_path = os.path.join(raw_wav_dir, spk, f"{spk}.txt")
    if not os.path.exists(txt_path):  # eval_full_testset.py: `if not os.path.exists(txt_path): continue`
        return {}
    with open(txt_path, "rb") as f:
        raw = f.read()
    # ESD's per-speaker txt files are inconsistently encoded (utf-8, utf-16 with BOM,
    # and at least one single-byte latin-1/cp1252 file). Detect by BOM first: Python's
    # incremental utf-16 decoder raises a bare UnicodeError ("UTF-16 stream does not
    # start with BOM") on BOM-less input, which an `except UnicodeDecodeError` chain
    # (eval_full_testset.load_reference_texts) does not catch.
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeError:
            text = raw.decode("latin-1")
    lines = text.splitlines(keepends=True)
    rows = {}
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            rows[parts[0]] = (parts[1], parts[2].strip())
        elif len(parts) == 2:
            rows[parts[0]] = (parts[1], None)
    return rows


def load_reference_texts(dataset, ids):
    """id -> reference transcript. Mirrors eval_full_testset.load_reference_texts
    (LJSpeech metadata.csv col 3; LibriTTS *.normalized.txt; ESD per-speaker txt)."""
    spec = DATASET_SPECS[dataset]
    texts = {}
    if dataset == "ljspeech":
        ids_set = set(ids)
        with open(spec["metadata_csv"], encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
                if row and row[0] in ids_set:
                    texts[row[0]] = row[2] if len(row) > 2 else row[1]
    elif dataset in ("libritts", "libritts_test_other"):
        for uid in ids:
            m = glob.glob(os.path.join(spec["raw_wav_dir"], "*", "*", f"{uid}.normalized.txt"))
            if m:
                with open(m[0], encoding="utf-8") as f:
                    texts[uid] = f.read().strip()
    elif dataset == "esd":
        ids_set = set(ids)
        for spk in sorted({u.split("_")[0] for u in ids}):
            for stem, (txt, _emo) in read_esd_speaker_txt(spec["raw_wav_dir"], spk).items():
                if stem in ids_set:
                    texts[stem] = txt
    return texts


def load_manifest(dataset):
    with open(os.path.join(MANIFEST_DIR, f"{dataset}.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]
