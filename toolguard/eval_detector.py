"""Run the off-the-shelf protectai DeBERTa detector: default (truncated at 512) and with our windowing."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import load
from .model import score_documents

MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    pos = [i for i, lab in model.config.id2label.items() if "INJ" in lab.upper()][0]
    out = {"model": MODEL_ID, "id2label": model.config.id2label}
    for split in ["test"]:
        texts = [r["text"] for r in load(split)]
        for name, windowed, ml in [("default_truncate512", False, 512), ("windowed512", True, 512)]:
            t0 = time.time()
            z, spans = score_documents(model, tok, texts, windowed=windowed, max_length=ml, positive_index=pos, batch_size=16)
            out[f"{split}_{name}"] = {"logit": z.tolist(), "span": spans, "seconds": time.time() - t0}
            print(split, name, f"{time.time() - t0:.0f}s", flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/raw_deberta_scores.json").write_text(json.dumps(out))
    print("mean", np.mean(out["test_default_truncate512"]["logit"]))


if __name__ == "__main__":
    main()
