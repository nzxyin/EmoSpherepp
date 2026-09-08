"""Per-group breakdown of a scored results tag (default: ESD by emotion and by
seen/unseen speaker): n, corpus WER raw and Whisper-normalized, UTMOSv2, DNSMOS
(ovr, p808), ECAPA speaker cosine, emotion2vec+ large emotion cosine (from
emotion_cosine_per_utt.py's dump when present). Same definitions/aggregation as
score_wav_pairs.py (corpus-level jiwer.wer; mean +/- 1.96*std/sqrt(n) for the
rest). Writes results/eval_<tag>_breakdown.json and a markdown table.
Run under the eval-articulatory-tts venv (transformers for the normalizer, jiwer).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_DIR, UNSEEN_SPEAKERS, load_manifest, read_jsonl  # noqa: E402

EMOTIONS = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]


def summarize(values):
    values = np.array([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    return {"mean": float(values.mean()), "ci95": float(1.96 * values.std() / np.sqrt(len(values))), "n": int(len(values))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="esd")
    ap.add_argument("--dataset", default="esd")
    ap.add_argument("--results_dir", default=RESULTS_DIR)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    import jiwer
    from transformers import WhisperTokenizer
    norm = WhisperTokenizer.from_pretrained("openai/whisper-large-v3").normalize

    partial = os.path.join(args.results_dir, f"eval_{args.tag}.json.partial.jsonl")
    recs = {}
    for d in read_jsonl(partial):
        recs[d.pop("uid")] = d
    meta = {r["uid"]: r for r in load_manifest(args.dataset)}
    emo_path = os.path.join(args.results_dir, f"eval_{args.tag}_emotion_per_utt.json")
    emo_cos = json.load(open(emo_path)) if os.path.exists(emo_path) else {}
    if not emo_cos:
        print(f"(no per-utterance emotion cosine at {emo_path}; emotion_cosine column will be n/a)")

    def group_metrics(uids):
        rs = [recs[u] for u in uids if u in recs]
        out = {"n": len(rs)}
        refs = [r["wer_reference"] for r in rs if "wer_hypothesis" in r]
        hyps = [r["wer_hypothesis"] for r in rs if "wer_hypothesis" in r]
        out["wer"] = {"mean": jiwer.wer(refs, hyps) if refs else None, "ci95": None, "n": len(refs)}
        npairs = [(r.get("wer_reference_normalized") or norm(r["wer_reference"]),
                   r.get("wer_hypothesis_normalized") or norm(r["wer_hypothesis"])) for r in rs if "wer_hypothesis" in r]
        npairs = [(a, b) for a, b in npairs if a.strip()]
        out["wer_whisper_normalized"] = {"mean": jiwer.wer([a for a, _ in npairs], [b for _, b in npairs]) if npairs else None,
                                         "ci95": None, "n": len(npairs)}
        for k in ("utmosv2", "dnsmos_ovr", "dnsmos_p808", "speaker_cosine"):
            out[k] = summarize([r.get(k) for r in rs])
        out["emotion_cosine"] = summarize([emo_cos.get(u) for u in uids if u in emo_cos])
        out["pred_gt_dur_ratio"] = summarize([r["pred_sec"] / r["gt_sec"] for r in rs if r.get("gt_sec")])
        return out

    uids_all = [u for u in meta if u in recs]
    groups = {"all": uids_all}
    for e in EMOTIONS:
        groups[e] = [u for u in uids_all if meta[u]["emotion"] == e]
    groups["seen speakers"] = [u for u in uids_all if meta[u]["spk"] not in UNSEEN_SPEAKERS]
    groups["unseen speakers (0013, 0019)"] = [u for u in uids_all if meta[u]["spk"] in UNSEEN_SPEAKERS]
    for e in EMOTIONS:
        groups[f"{e} / seen"] = [u for u in groups[e] if meta[u]["spk"] not in UNSEEN_SPEAKERS]
        groups[f"{e} / unseen"] = [u for u in groups[e] if meta[u]["spk"] in UNSEEN_SPEAKERS]
    result = {g: group_metrics(u) for g, u in groups.items()}

    out_json = os.path.join(args.results_dir, f"eval_{args.tag}_breakdown.json")
    with open(out_json + ".tmp", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(out_json + ".tmp", out_json)

    def cell(m, key, scale=1.0, fmt="{:.3f}", ci=True):
        v = m[key]
        if v["mean"] is None:
            return "n/a"
        s = fmt.format(v["mean"] * scale)
        if ci and v.get("ci95") is not None:
            s += " ±" + fmt.format(v["ci95"] * scale).rstrip("%")
        return s

    def table(names, title):
        lines = [f"**{title}**", "", "| group | n | WER raw | WER-n | UTMOSv2 | DNSMOS_ovr | DNSMOS_p808 | speaker_cos | emotion_cos | pred/GT dur |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for g in names:
            m = result[g]
            lines.append(f"| {g} | {m['n']} | {cell(m, 'wer', 100, '{:.2f}%')} | {cell(m, 'wer_whisper_normalized', 100, '{:.2f}%')} | "
                         f"{cell(m, 'utmosv2')} | {cell(m, 'dnsmos_ovr')} | {cell(m, 'dnsmos_p808')} | {cell(m, 'speaker_cosine')} | "
                         f"{cell(m, 'emotion_cosine')} | {cell(m, 'pred_gt_dur_ratio', 1.0, '{:.2f}', ci=False)} |")
        return "\n".join(lines)

    md = [f"# {args.tag}: breakdown (generated by eval/breakdown.py)", "",
          table(["all"] + EMOTIONS, "Per emotion (all 10 speakers)"), "",
          table(["seen speakers", "unseen speakers (0013, 0019)"], "Seen vs unseen speakers"), "",
          table([f"{e} / seen" for e in EMOTIONS] + [f"{e} / unseen" for e in EMOTIONS], "Emotion x speaker subset")]
    md = "\n".join(md) + "\n"
    print(md)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, "w") as f:
            f.write(md)
        print(f"wrote {args.out_md}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
