"""In-distribution check: how well do the trained models do on validation (same sources as train)?
Separates 'did not learn' from 'learned something that does not transfer to unseen attack authors'."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .baselines import tfidf_score, train_tfidf
from .data import load
from .evaluate import load_tg
from .model import score_documents


def main():
    torch.set_num_threads(8)
    train, val, test = load("train"), load("val"), load("test")
    yv = np.array([r["label"] for r in val])
    pipe = train_tfidf([r["text"] for r in train], [r["label"] for r in train])
    out = {"val_n": len(val), "val_positive_rate": float(yv.mean())}
    out["tfidf_val_auroc"] = float(roc_auc_score(yv, tfidf_score(pipe, [r["text"] for r in val])))
    tok, m, cfg = load_tg("models/toolguard-minilm")
    z, _ = score_documents(m, tok, [r["text"] for r in val])
    out["toolguard_val_auroc"] = float(roc_auc_score(yv, z))
    # raw (un-embedded) attacks vs raw benign prompts in validation, and raw Gandalf attacks alone
    raw = [r for r in val if r["fmt"] == "raw"]
    zr, _ = score_documents(m, tok, [r["text"] for r in raw])
    out["toolguard_val_raw_prompts_auroc"] = float(roc_auc_score([r["label"] for r in raw], zr))
    atk = sorted({r["attack"] for r in test if r.get("attack")})
    za, _ = score_documents(m, tok, atk)
    p = 1 / (1 + np.exp(-za / cfg["temperature"]))
    out["toolguard_recall_on_bare_gandalf_attacks"] = float((p >= cfg["threshold"]).mean())
    out["bare_gandalf_attacks_n"] = len(atk)
    Path("results/val_sanity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
