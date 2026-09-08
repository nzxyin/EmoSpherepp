"""Score a directory of {uid}_pred.wav / {uid}_gt.wav pairs with the SAME metric
stack, models and aggregation as articulatory-tts/eval_full_testset.py's score()
(copied 2026-09-07; that script fuses prediction and scoring around its own SPARC
model, so for an external TTS system the scoring half is reproduced here):

  wer            Whisper large-v3 (transformers, float32, language=en, task=transcribe),
                 lowercased ref/hyp, CORPUS-level jiwer.wer over the whole set (ci95=None) --
                 the raw convention articulatory-tts used until 2026-09-08 (GH #32)
  wer_whisper_normalized  same pairs after whisper_processor.tokenizer.normalize() on both
                 sides (Whisper's EnglishTextNormalizer: punctuation, casing, numbers, spelling),
                 pairs with an empty normalized reference skipped -- articulatory-tts's WER
                 since its GH #32 fix, and the TTS repo's extra key of the same name
  utmosv2        utmosv2.create_model(pretrained=True).predict(...)  (16 kHz)
  dnsmos_*       torchmetrics DNSMOS (personalized=False): p808, sig, bak, ovr
  speaker_cosine ECAPA-TDNN speechbrain/spkrec-ecapa-voxceleb encode_batch cosine, pred vs GT,
                 skipped when either clip < 0.3 s (MIN_ECAPA_SEC)
  emotion_cosine NOT here -- run articulatory-tts/score_side_metric.py --metric emotion in its
                 eval-emotion venv on the same wav_pairs dir, then merge_eval_results.py.

Reference transcripts come from the same corpus files (eval/common.load_reference_texts).
Run under /data/user_data/xoy/venvs/eval-articulatory-tts (not this repo's venv).
Output JSON has eval_full_testset.py's schema so the same downstream tooling applies.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_reference_texts  # noqa: E402

MIN_ECAPA_SEC = 0.3


def summarize(values):
    values = np.array(values)
    if len(values) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    return {"mean": float(values.mean()), "ci95": float(1.96 * values.std() / np.sqrt(len(values))), "n": len(values)}


def find_pairs(wav_pairs_dir):
    pairs = []
    for pred_path in sorted(glob.glob(os.path.join(wav_pairs_dir, "*_pred.wav"))):
        uid = os.path.basename(pred_path)[: -len("_pred.wav")]
        gt_path = os.path.join(wav_pairs_dir, f"{uid}_gt.wav")
        if os.path.exists(gt_path):
            pairs.append((uid, pred_path, gt_path))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["ljspeech", "libritts", "libritts_test_other", "esd"])
    ap.add_argument("--wav_pairs_dir", required=True)
    ap.add_argument("--results_path", required=True)
    ap.add_argument("--per_utt_out", default=None)
    ap.add_argument("--partial_path", default=None,
                    help="per-utterance checkpoint JSONL (default <results_path>.partial.jsonl): every scored utterance is "
                         "appended immediately and skipped on rerun, so a preempted/requeued job resumes instead of redoing Whisper")
    ap.add_argument("--system", default="EmoSpherepp model_ckpt_steps_1100000 + speechbrain hifigan-libritts-16kHz")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip_audio", action="store_true")
    ap.add_argument("--skip_wer", action="store_true")
    ap.add_argument("--skip_speaker", action="store_true")
    args = ap.parse_args()

    import soundfile as sf
    import librosa

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = find_pairs(args.wav_pairs_dir)
    if args.limit:
        pairs = pairs[: args.limit]
    ids = [p[0] for p in pairs]
    print(f"Scoring {len(pairs)} {args.dataset} pred/gt pairs from {args.wav_pairs_dir} (device={device})")
    if not pairs:
        raise RuntimeError("no pairs found")
    from common import DATASET_SPECS, MANIFEST_DIR, load_manifest
    split_file = os.path.basename(DATASET_SPECS[args.dataset]["split_path"])
    try:
        n_manifest = len(load_manifest(args.dataset))
    except FileNotFoundError:
        n_manifest = None
    # eval_full_testset.py reports n_utterances = len(ids) of the split file; prepare_manifests.py
    # records that in <dataset>.stats.json (it drops ids with no wav/text, so len(manifest) can be smaller).
    n_split = n_manifest
    stats_path = os.path.join(MANIFEST_DIR, f"{args.dataset}.stats.json")
    if os.path.exists(stats_path):
        n_split = json.load(open(stats_path)).get("n_ids", n_manifest)
    if n_manifest is not None and not args.limit and len(pairs) != n_manifest:
        print(f"WARNING: {len(pairs)} wav pairs on disk but {n_manifest} utterances in the {args.dataset} manifest "
              f"({n_manifest - len(pairs)} missing) -- results cover the pairs present only")

    dnsmos_fn = utmos_model = whisper_processor = whisper_model = ecapa_model = None
    if not args.skip_audio:
        from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score as dnsmos_fn
        import utmosv2
        utmos_model = utmosv2.create_model(pretrained=True)
    if not args.skip_wer:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3", torch_dtype=torch.float32).to(device).eval()
        whisper_normalize = whisper_processor.tokenizer.normalize
        import jiwer
    if not args.skip_speaker:
        from speechbrain.inference.speaker import EncoderClassifier
        ecapa_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(os.environ.get("HF_HOME", "/tmp"), "spkrec-ecapa-voxceleb"),
            run_opts={"device": str(device)},
        )

    def to16k(wav, sr):
        return wav if sr == 16000 else librosa.resample(wav, orig_sr=sr, target_sr=16000)

    def dnsmos_score(wav, sr):
        wav = to16k(wav, sr)
        p808, sig, bak, ovr = dnsmos_fn(torch.from_numpy(wav).float().to(device), 16000, personalized=False, num_threads=4)
        return {"p808": float(p808), "sig": float(sig), "bak": float(bak), "ovr": float(ovr)}

    def utmos_score(wav, sr):
        result = utmos_model.predict(data=to16k(wav, sr).astype(np.float32), sr=16000)
        return float(np.asarray(result).reshape(-1)[0])

    def whisper_transcribe(wav, sr):
        inputs = whisper_processor(to16k(wav, sr), sampling_rate=16000, return_tensors="pt").input_features.to(device)
        ids_out = whisper_model.generate(inputs, language="en", task="transcribe")
        return whisper_processor.batch_decode(ids_out, skip_special_tokens=True)[0].strip()

    def ecapa_embed(wav, sr):
        wav_tensor = torch.from_numpy(to16k(wav, sr)).float().unsqueeze(0).to(device)
        return ecapa_model.encode_batch(wav_tensor).squeeze().detach().cpu().numpy().reshape(-1)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    reference_texts = {} if args.skip_wer else load_reference_texts(args.dataset, ids)
    if not args.skip_wer:
        missing = set(ids) - set(reference_texts)
        if missing:
            print(f"WARNING: no reference text for {len(missing)}/{len(ids)} utterances (e.g. {sorted(missing)[:3]})")

    from tqdm import tqdm
    # Per-utterance checkpointing (preempt-tier jobs get evicted mid-run): resume from
    # the partial JSONL, skipping utterances whose record already holds every metric
    # this invocation would compute.
    partial_path = args.partial_path or (args.results_path + ".partial.jsonl")
    per_utt_records = {}
    if os.path.exists(partial_path):
        from common import read_jsonl
        for rec in read_jsonl(partial_path):
            per_utt_records[rec.pop("uid")] = rec

    def complete(rec):
        if not args.skip_audio and "utmosv2" not in rec:
            return False
        if not args.skip_wer and "wer_hypothesis" not in rec and rec.get("_wer_skipped") is not True:
            return False
        if not args.skip_speaker and "speaker_cosine" not in rec and rec.get("_speaker_skipped") is not True:
            return False
        return True

    todo = [(u, p, g) for u, p, g in pairs if not (u in per_utt_records and complete(per_utt_records[u]))]
    print(f"{len(pairs) - len(todo)} utterances already scored in {partial_path}; scoring {len(todo)}")
    os.makedirs(os.path.dirname(os.path.abspath(partial_path)), exist_ok=True)
    partial_f = open(partial_path, "a")
    n_err = 0
    for uid, pred_path, gt_path in tqdm(todo, desc="scoring"):
        try:
            pred_wav, pred_sr = sf.read(pred_path)
            pred_wav = pred_wav.astype(np.float32)
            gt_wav, gt_sr = sf.read(gt_path)
            gt_wav = gt_wav.astype(np.float32)
            if pred_wav.ndim > 1:
                pred_wav = pred_wav.mean(axis=1)
            if gt_wav.ndim > 1:
                gt_wav = gt_wav.mean(axis=1)
            rec = per_utt_records.setdefault(uid, {})
            rec.update({"pred_sec": len(pred_wav) / pred_sr, "gt_sec": len(gt_wav) / gt_sr})

            if not args.skip_audio and "utmosv2" not in rec:
                d = dnsmos_score(pred_wav, pred_sr)
                rec.update({f"dnsmos_{k}": v for k, v in d.items()})
                rec["utmosv2"] = utmos_score(pred_wav, pred_sr)

            if not args.skip_wer and "wer_hypothesis" not in rec:
                if uid in reference_texts:
                    hyp = whisper_transcribe(pred_wav, pred_sr)
                    ref_lower, hyp_lower = reference_texts[uid].lower(), hyp.lower()
                    rec.update({"wer": jiwer.wer(ref_lower, hyp_lower), "wer_reference": ref_lower, "wer_hypothesis": hyp_lower,
                                "wer_reference_normalized": whisper_normalize(ref_lower),
                                "wer_hypothesis_normalized": whisper_normalize(hyp_lower)})
                else:
                    rec["_wer_skipped"] = True  # no reference text -> excluded from WER only (as in eval_full_testset.py)

            if not args.skip_speaker and "speaker_cosine" not in rec:
                pred_sec, gt_sec = len(pred_wav) / pred_sr, len(gt_wav) / gt_sr
                if pred_sec < MIN_ECAPA_SEC or gt_sec < MIN_ECAPA_SEC:
                    tqdm.write(f"WARNING: skipping speaker_cosine for {uid} -- too short (pred={pred_sec:.3f}s, gt={gt_sec:.3f}s)")
                    rec["_speaker_skipped"] = True
                else:
                    rec["speaker_cosine"] = cosine(ecapa_embed(pred_wav, pred_sr), ecapa_embed(gt_wav, gt_sr))

            partial_f.write(json.dumps({"uid": uid, **rec}) + "\n")
            partial_f.flush()
        except Exception as e:  # noqa: BLE001 -- one bad file must not kill a multi-hour job; it is reported below
            n_err += 1
            per_utt_records.pop(uid, None)
            tqdm.write(f"ERROR scoring {uid}: {type(e).__name__}: {e}")
    partial_f.close()
    if n_err:
        print(f"WARNING: {n_err} utterances failed to score (excluded from every metric)")

    # Aggregate over ALL scored utterances (resumed + new), in test-set order -- the same
    # aggregation eval_full_testset.py applies to its in-memory per_utt lists.
    recs = [per_utt_records[u] for u, _p, _g in pairs if u in per_utt_records]
    summary = {}
    if not args.skip_audio:
        for k in ("p808", "sig", "bak", "ovr"):
            summary[f"dnsmos_{k}"] = summarize([r[f"dnsmos_{k}"] for r in recs if f"dnsmos_{k}" in r])
        summary["utmosv2"] = summarize([r["utmosv2"] for r in recs if "utmosv2" in r])
    if not args.skip_wer:
        wer_refs = [r["wer_reference"] for r in recs if "wer_hypothesis" in r]
        wer_hyps = [r["wer_hypothesis"] for r in recs if "wer_hypothesis" in r]
        summary["wer"] = {"mean": jiwer.wer(wer_refs, wer_hyps) if wer_refs else None, "ci95": None, "n": len(wer_refs)}
        # Normalized variant (records from older runs lack the *_normalized keys; eval/normalized_wer.py
        # back-fills wer_whisper_normalized for those from the stored raw text).
        npairs = [(r["wer_reference_normalized"], r["wer_hypothesis_normalized"]) for r in recs
                  if "wer_reference_normalized" in r and r["wer_reference_normalized"].strip()]
        if npairs:
            summary["wer_whisper_normalized"] = {"mean": jiwer.wer([a for a, _ in npairs], [b for _, b in npairs]),
                                                 "ci95": None, "n": len(npairs)}
    if not args.skip_speaker:
        summary["speaker_cosine"] = summarize([r["speaker_cosine"] for r in recs if "speaker_cosine" in r])
    n_scored = len(recs)
    if n_scored != len(pairs):
        print(f"WARNING: {len(pairs) - n_scored} pairs have no record (errors?)")

    if args.per_utt_out:
        with open(args.per_utt_out + ".tmp", "w") as f:
            json.dump(per_utt_records, f, indent=1)
        os.replace(args.per_utt_out + ".tmp", args.per_utt_out)
        print(f"Wrote per-utterance metrics for {len(per_utt_records)} utterances to {args.per_utt_out}")

    results = {"checkpoint": args.system, "dataset": args.dataset, "split": split_file,
               "tag": os.path.basename(args.wav_pairs_dir), "wav_pairs_dir": args.wav_pairs_dir,
               "n_utterances": n_split if n_split is not None else len(pairs),  # split-file size, as in eval_full_testset.py
               "n_manifest": n_manifest, "n_pairs": len(pairs), "n_predicted": n_scored, "metrics": summary}
    print(json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(args.results_path)), exist_ok=True)
    with open(args.results_path + ".tmp", "w") as f:
        json.dump(results, f, indent=2)
    os.replace(args.results_path + ".tmp", args.results_path)  # atomic: run_score.sh's skip check must never see a torn file
    print(f"\nWrote results to {args.results_path}")


if __name__ == "__main__":
    main()
