"""Quick look at results/: headline metrics per results JSON, pred/GT duration
ratio, and the worst-WER utterances (from the per-utterance partial JSONL).
Pure-stdlib so it runs with any python3 on a node with /data mounted.
Usage: inspect_results.py [results_dir] [tags...]"""
import glob
import json
import os
import statistics as st
import sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "/data/user_data/xoy/EmoSpherepp_eval/results"
TAGS = sys.argv[2:]


def fmt(m, key, scale=1.0, f="{:.3f}"):
    v = m.get(key)
    if not v or v.get("mean") is None:
        return "n/a"
    s = f.format(v["mean"] * scale)
    if v.get("ci95") is not None:
        s += " +/-" + f.format(v["ci95"] * scale).rstrip("%")
    return s + " (n=%d)" % v["n"]


paths = sorted(glob.glob(os.path.join(RESULTS, "eval_*.json")))
paths = [p for p in paths if not p.endswith(("_per_utt.json", "_emotion.json"))]
for p in paths:
    tag = os.path.basename(p)[5:-5]
    if TAGS and tag not in TAGS:
        continue
    d = json.load(open(p))
    m = d.get("metrics", {})
    print("== %s: n_utterances=%s n_pairs=%s n_predicted=%s" % (tag, d.get("n_utterances"), d.get("n_pairs"), d.get("n_predicted")))
    print("   WER %s | UTMOSv2 %s | DNSMOS_ovr %s | DNSMOS_p808 %s" % (fmt(m, "wer", 100, "{:.2f}%"), fmt(m, "utmosv2"), fmt(m, "dnsmos_ovr"), fmt(m, "dnsmos_p808")))
    print("   speaker_cosine %s | emotion_cosine %s" % (fmt(m, "speaker_cosine"), fmt(m, "emotion_cosine")))
    partial = p + ".partial.jsonl"
    if os.path.exists(partial):
        recs = []
        for l in open(partial):
            if l.strip():
                try:
                    recs.append(json.loads(l))
                except json.JSONDecodeError:
                    print("   WARNING: dropping unparsable checkpoint line (job was likely preempted mid-write)")
        ratios = [r["pred_sec"] / r["gt_sec"] for r in recs if r.get("gt_sec")]
        if ratios:
            print("   pred/gt duration ratio: median=%.3f mean=%.3f min=%.2f max=%.2f (n=%d)" % (st.median(ratios), st.mean(ratios), min(ratios), max(ratios), len(ratios)))
        wers = [r for r in recs if "wer" in r]
        if wers:
            macro = st.mean(r["wer"] for r in wers)
            print("   per-utt macro WER=%.2f%% ; utts with WER>=0.5: %d" % (macro * 100, sum(r["wer"] >= 0.5 for r in wers)))
            for r in sorted(wers, key=lambda r: -r["wer"])[:3]:
                print("   worst wer=%.2f uid=%s gt=%.1fs pred=%.1fs" % (r["wer"], r["uid"], r["gt_sec"], r["pred_sec"]))
                print("      REF: %s" % r["wer_reference"][:120])
                print("      HYP: %s" % r["wer_hypothesis"][:120])
