"""Collect results/eval_<tag>.json files into one markdown table (same columns
articulatory-tts's CLAUDE.md tables use: WER, UTMOSv2, DNSMOS_ovr, speaker_cosine,
emotion_cosine; plus n). Usage: summarize_results.py [--results_dir DIR] [--tags a b ...]"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_DIR  # noqa: E402

COLS = [("wer", "WER", 100.0, "{:.2f}%"), ("utmosv2", "UTMOSv2", 1.0, "{:.3f}"), ("dnsmos_ovr", "DNSMOS_ovr", 1.0, "{:.3f}"),
        ("dnsmos_p808", "DNSMOS_p808", 1.0, "{:.3f}"), ("speaker_cosine", "speaker_cosine", 1.0, "{:.3f}"),
        ("emotion_cosine", "emotion_cosine", 1.0, "{:.3f}")]


def fmt(m, key, scale, f):
    v = m.get(key)
    if not v or v.get("mean") is None:
        return "n/a"
    s = f.format(v["mean"] * scale)
    if v.get("ci95") is not None:
        s += " ± " + f.format(v["ci95"] * scale).rstrip("%")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--tags", nargs="*", default=None)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.results_dir, "eval_*.json")))
    paths = [p for p in paths if not p.endswith(("_per_utt.json", "_emotion.json"))]
    if args.tags:
        paths = [p for p in paths if os.path.basename(p)[5:-5] in args.tags]
    print("| test set | n | " + " | ".join(c[1] for c in COLS) + " |")
    print("|" + "---|" * (len(COLS) + 2))
    for p in paths:
        d = json.load(open(p))
        m = d.get("metrics", {})
        n = (m.get("wer") or {}).get("n") or (m.get("utmosv2") or {}).get("n") or d.get("n_utterances")
        tag = os.path.basename(p)[5:-5]
        print(f"| {tag} | {n} | " + " | ".join(fmt(m, k, s, f) for k, _l, s, f in COLS) + " |")


if __name__ == "__main__":
    main()
