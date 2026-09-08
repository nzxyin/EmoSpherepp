"""Rebuild the phone_set.json the released checkpoint was trained with.

The binarized ESD data (which carried phone_set.json) was never released, but
the phone set is a deterministic function of the ESD transcripts shipped in
this repo (esd_text_emo.txt) and the repo's own English text front-end
(data_gen/tts/txt_processors/en.py: g2p_en + '|' word separators + <BOS>/<EOS>),
built as BasePreprocessor._phone_encoder does: sorted(set(all phonemes)).

Sanity check: TokenTextEncoder prepends the reserved tokens [<pad>, <EOS>, <UNK>]
(skipping <EOS> from the list since it's reserved), and the model's embedding
has len(encoder)+1 rows (the extra row is the intersperse blank token) -- the
checkpoint's encoder.emb.weight must match exactly or this is wrong.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))

from common import CKPT_PATH, PHONE_SET  # noqa: E402
from data_gen.tts.txt_processors.en import TxtProcessor  # noqa: E402
from utils.text.text_encoder import build_token_encoder  # noqa: E402

PREPROCESS_ARGS = {"txt_processor": "en", "with_phsep": True, "add_eos_bos": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esd_text", default=os.path.join(REPO, "esd_text_emo.txt"))
    ap.add_argument("--out", default=PHONE_SET)
    ap.add_argument("--ckpt", default=CKPT_PATH)
    args = ap.parse_args()

    phones = set()
    n_ok = n_fail = 0
    with open(args.esd_text, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _wav, txt, _emo = line.split("|")
            try:
                txt_struct, _ = TxtProcessor.process(txt, PREPROCESS_ARGS)
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"WARN g2p failed on {txt!r}: {e}")
                continue
            for _w, phs in txt_struct:
                phones.update(phs)
            n_ok += 1
    ph_set = sorted(phones)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(ph_set, open(args.out, "w"), ensure_ascii=False)
    print(f"processed {n_ok} lines ({n_fail} failed); {len(ph_set)} phones -> {args.out}")
    print(ph_set)
    enc = build_token_encoder(args.out)
    print(f"token encoder vocab_size={len(enc)} -> model n_vocab={len(enc) + 1}")

    if os.path.exists(args.ckpt):
        import torch
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
        if "model" in sd and isinstance(sd["model"], dict):  # nested {'model': {...}} layout
            sd = sd["model"]
        emb = [v for k, v in sd.items() if k.endswith("encoder.emb.weight")][0]
        print(f"checkpoint encoder.emb.weight: {tuple(emb.shape)}")
        assert emb.shape[0] == len(enc) + 1, (
            f"phone set mismatch: checkpoint expects n_vocab={emb.shape[0]}, rebuilt set gives {len(enc) + 1}")
        print("OK: rebuilt phone set matches the checkpoint's vocabulary size")
    else:
        print(f"(checkpoint not found at {args.ckpt}; skipped vocab-size check)")


if __name__ == "__main__":
    main()
