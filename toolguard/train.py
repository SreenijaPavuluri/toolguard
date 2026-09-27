"""Fine-tune all-MiniLM-L6-v2 as a window-level injection classifier on CPU, then calibrate."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import load
from .model import score_documents
from .windowing import make_windows

BASE = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 256


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def window_examples(rows, tok, rng, max_windows):
    """Turn documents into labelled 256-token windows. The window that contains the inserted text
    gets the document label, and one other window from the same document is added as a negative."""
    ex = []
    for r in rows:
        ws = make_windows(r["text"], tok, MAX_LEN, 64)
        span = r.get("span")
        if span:
            ov = [max(0, min(w.char_end, span[1]) - max(w.char_start, span[0])) for w in ws]
            k = int(np.argmax(ov))
            ex.append((ws[k].input_ids, r["label"]))
            others = [w for w, o in zip(ws, ov) if o == 0]
            if others:
                ex.append((rng.choice(others).input_ids, 0))
        else:
            ex.append((rng.choice(ws).input_ids, r["label"]))
    rng.shuffle(ex)
    return ex[:max_windows]


def fit_temperature(z, y):
    z, y = torch.tensor(z, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    z = z.clamp(-50, 50)
    logt = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / logt.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logt.exp())


def calibrate(model, tok, val_rows, target_fpr=0.01):
    z, _ = score_documents(model, tok, [r["text"] for r in val_rows], windowed=True, max_length=MAX_LEN)
    y = np.array([r["label"] for r in val_rows])
    T = fit_temperature(z, y)
    p = 1 / (1 + np.exp(-np.clip(z / T, -50, 50)))
    neg = np.sort(p[y == 0])
    k = int(np.floor(target_fpr * len(neg)))
    thr = float(neg[len(neg) - 1 - k])
    return T, thr, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train")
    ap.add_argument("--out", default="models/toolguard-minilm")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-windows", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_seed(a.seed)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, id2label={0: "SAFE", 1: "INJECTION"},
                                                               label2id={"SAFE": 0, "INJECTION": 1})
    ex = window_examples(load(a.train), tok, random.Random(a.seed), a.max_windows)
    print(f"{len(ex)} windows, positive rate {np.mean([e[1] for e in ex]):.3f}", flush=True)
    bs = 16
    steps = int(np.ceil(len(ex) / bs) * a.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train()
    t0 = time.time()
    for step in range(steps):
        i = (step * bs) % len(ex)
        batch = ex[i:i + bs]
        m = max(len(e[0]) for e in batch)
        ids = torch.zeros((len(batch), m), dtype=torch.long)
        att = torch.zeros((len(batch), m), dtype=torch.long)
        for r, (w, _) in enumerate(batch):
            ids[r, :len(w)] = torch.tensor(w)
            att[r, :len(w)] = 1
        y = torch.tensor([e[1] for e in batch])
        loss = model(input_ids=ids, attention_mask=att, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        opt.zero_grad()
        if step % 10 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    T, thr, _ = calibrate(model, tok, load("val"))
    out = Path(a.out)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    cfg = {"base_model": BASE, "max_length": MAX_LEN, "stride": 64, "temperature": T, "threshold": thr,
           "threshold_rule": "1% FPR on validation documents", "train_file": a.train, "train_windows": len(ex),
           "epochs": a.epochs, "seed": a.seed, "train_seconds": time.time() - t0}
    (out / "toolguard_config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg), flush=True)


if __name__ == "__main__":
    main()
