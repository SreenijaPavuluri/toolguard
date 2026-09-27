"""Window-level scoring shared by ToolGuard and the off-the-shelf detector wrapper."""
from __future__ import annotations

import numpy as np
import torch

from .windowing import make_windows, truncate_window


def _batch_logits(model, windows, pad_id, batch_size=32):
    out = []
    order = np.argsort([len(w.input_ids) for w in windows])  # length-sorted batches = less padding
    res = [None] * len(windows)
    with torch.inference_mode():
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            m = max(len(windows[j].input_ids) for j in idx)
            ids = torch.full((len(idx), m), pad_id, dtype=torch.long)
            att = torch.zeros((len(idx), m), dtype=torch.long)
            for r, j in enumerate(idx):
                w = windows[j].input_ids
                ids[r, :len(w)] = torch.tensor(w)
                att[r, :len(w)] = 1
            logits = model(input_ids=ids, attention_mask=att).logits.float()
            for r, j in enumerate(idx):
                res[j] = logits[r]
    out = torch.stack(res) if res else torch.zeros((0, 2))
    return out


def injection_logit(logits, positive_index=1):
    """Binary log-odds for the injection class from 2-class logits."""
    return (logits[:, positive_index] - logits[:, 1 - positive_index]).numpy()


def score_documents(model, tokenizer, texts, windowed=True, max_length=256, stride=64, positive_index=1, batch_size=32):
    """Return per-document (max log-odds, char span of the top window)."""
    wins, owner = [], []
    for d, t in enumerate(texts):
        ws = make_windows(t, tokenizer, max_length, stride) if windowed else [truncate_window(t, tokenizer, max_length)]
        wins += ws
        owner += [d] * len(ws)
    z = injection_logit(_batch_logits(model, wins, tokenizer.pad_token_id, batch_size), positive_index)
    best = np.full(len(texts), -np.inf)
    spans = [None] * len(texts)
    for k, d in enumerate(owner):
        if z[k] > best[d]:
            best[d] = z[k]
            spans[d] = (wins[k].char_start, wins[k].char_end)
    return best, spans
