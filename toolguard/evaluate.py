"""Evaluate every model and ablation on the cross-source test set. Writes results/*.json and charts."""
from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .baselines import regex_score, tfidf_score, train_tfidf
from .data import DEPTHS, LENGTHS, load
from .metrics import bootstrap, rate_ci
from .model import score_documents

R = Path("results")


def _quantize(model, *args, **kwargs):
    """Dynamic int8 quantization; picks an available engine (qnnpack on ARM, fbgemm on x86)."""
    import torch
    engines = torch.backends.quantized.supported_engines
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in engines else "fbgemm"
    return torch.ao.quantization.quantize_dynamic(model, *args, **kwargs)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def load_tg(path):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).eval()
    cfg = json.loads((Path(path) / "toolguard_config.json").read_text())
    return tok, m, cfg


def dir_mb(path):
    return sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 1e6


def main(latency_docs=10):
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("TG_THREADS", "8")))
    test, train = load("test"), load("train")
    texts = [r["text"] for r in test]
    y = np.array([r["label"] for r in test])
    models = {}  # name -> dict(score, pred, span)

    # 1. regex
    s = np.array([regex_score(t) for t in texts], dtype=float)
    models["Regex rules"] = dict(score=s, pred=s > 0)
    # 2. tf-idf + LR on whole documents
    pipe = train_tfidf([r["text"] for r in train], [r["label"] for r in train])
    s = tfidf_score(pipe, texts)
    models["TF-IDF + LR"] = dict(score=s, pred=s >= 0.5)
    # 3. off-the-shelf detector (scores produced by toolguard.eval_detector)
    raw = json.loads((R / "raw_deberta_scores.json").read_text())
    for key, name in [("test_default_truncate512", "ProtectAI DeBERTa (default, truncate 512)"),
                      ("test_windowed512", "ProtectAI DeBERTa + our windowing")]:
        if key in raw:
            z = np.array(raw[key]["logit"])
            models[name] = dict(score=z, pred=z > 0, span=raw[key]["span"])
    # 4. ToolGuard and ablations
    tok, tg, cfg = load_tg("models/toolguard-minilm")
    T, thr = cfg["temperature"], cfg["threshold"]
    z, spans = score_documents(tg, tok, texts, windowed=True)
    p = sigmoid(z / T)
    models["ToolGuard (MiniLM, windowed)"] = dict(score=p, pred=p >= thr, span=spans)
    z2, _ = score_documents(tg, tok, texts, windowed=False)
    p2 = sigmoid(z2 / T)
    models["Ablation: ToolGuard without windowing"] = dict(score=p2, pred=p2 >= thr)
    if Path("models/toolguard-minilm-nohn").exists():
        tok3, tg3, cfg3 = load_tg("models/toolguard-minilm-nohn")
        z3, _ = score_documents(tg3, tok3, texts, windowed=True)
        p3 = sigmoid(z3 / cfg3["temperature"])
        models["Ablation: ToolGuard without hard negatives"] = dict(score=p3, pred=p3 >= cfg3["threshold"])
    tgq = _quantize(tg, {torch.nn.Linear}, dtype=torch.qint8)
    zq, _ = score_documents(tgq, tok, texts, windowed=True)
    pq = sigmoid(zq / T)
    models["ToolGuard int8 (dynamic quantization)"] = dict(score=pq, pred=pq >= thr)

    # ------------------------------------------------ main table
    src = np.array([r["source"] for r in test])
    hn, clean, pos = src == "real_hard_negative", src == "carrier_clean", y == 1
    main_tab = {}
    for name, m in models.items():
        res = bootstrap(y, m["score"], m["pred"])
        res["fpr_real_hard_negatives"] = rate_ci(np.asarray(m["pred"])[hn].astype(float))
        res["fpr_clean_carriers"] = rate_ci(np.asarray(m["pred"])[clean].astype(float))
        # AUROC with only real hard negatives as the negative class
        res["auroc_vs_hard_negatives"] = bootstrap(y[pos | hn], np.asarray(m["score"])[pos | hn], np.asarray(m["pred"])[pos | hn])["auroc"]
        main_tab[name] = res
    (R / "main_results.json").write_text(json.dumps(main_tab, indent=2))

    # ------------------------------------------------ dilution
    dil = {"by_depth": {}, "by_length": {}, "by_format": {}}
    depth = np.array([r["depth"] if r["depth"] is not None else -1 for r in test], dtype=float)
    length = np.array([r["length"] for r in test])
    fmt = np.array([r["fmt"] for r in test])
    for name, m in models.items():
        pr = np.asarray(m["pred"]).astype(float)
        dil["by_depth"][name] = {str(d): rate_ci(pr[pos & (depth == d)]) for d in DEPTHS}
        dil["by_length"][name] = {str(L): rate_ci(pr[pos & (length == L)]) for L in LENGTHS}
        dil["by_format"][name] = {f: rate_ci(pr[pos & (fmt == f)]) for f in ["plain", "html", "markdown", "json"]}
    # depth x length for the off-the-shelf detector: the dilution mechanism is truncation
    tok_counts = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    dil["minilm_token_count_by_length_bin"] = {str(L): statistics.median([c for c, lb, yy in zip(tok_counts, length, y) if lb == L and yy == 1]) for L in LENGTHS}
    (R / "dilution.json").write_text(json.dumps(dil, indent=2))

    # ------------------------------------------------ span localisation
    span_hits = []
    for r, sp, pr in zip(test, spans, models["ToolGuard (MiniLM, windowed)"]["pred"]):
        if r["label"] == 1 and pr:
            a, b = r["span"]
            span_hits.append(float(sp[0] <= a and b <= sp[1]))
    (R / "span_localisation.json").write_text(json.dumps({"detected_positives": len(span_hits),
        "fraction_top_window_contains_full_attack": float(np.mean(span_hits)) if span_hits else None}, indent=2))

    # ------------------------------------------------ latency and size
    rng = np.random.default_rng(0)
    idx = rng.choice(len(texts), latency_docs, replace=False)
    lat_docs = [texts[i] for i in idx]
    lat = {}

    def timeit(fn):
        fn(lat_docs[:2])  # warm-up
        ts = []
        for d in lat_docs:
            t0 = time.perf_counter()
            fn([d])
            ts.append((time.perf_counter() - t0) * 1000)
        return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "docs": len(ts),
                "median_minilm_tokens": float(np.median([tok_counts[i] for i in idx]))}

    lat["Regex rules"] = {**timeit(lambda d: [regex_score(t) for t in d]), "size_mb": 0.0}
    lat["TF-IDF + LR"] = {**timeit(lambda d: tfidf_score(pipe, d)), "size_mb": pipe[-1].coef_.nbytes / 1e6 + len(pipe[0].vocabulary_) * 40 / 1e6}
    lat["ToolGuard (MiniLM, windowed)"] = {**timeit(lambda d: score_documents(tg, tok, d)), "size_mb": dir_mb("models/toolguard-minilm")}
    lat["Ablation: ToolGuard without windowing"] = {**timeit(lambda d: score_documents(tg, tok, d, windowed=False)), "size_mb": dir_mb("models/toolguard-minilm")}
    qsize = sum(t.numel() * t.element_size() for t in tgq.state_dict().values() if torch.is_tensor(t)) / 1e6
    lat["ToolGuard int8 (dynamic quantization)"] = {**timeit(lambda d: score_documents(tgq, tok, d)), "size_mb_state_dict_estimate": qsize}
    if os.environ.get("TG_SKIP_DEBERTA_LATENCY") != "1":
        from .eval_detector import MODEL_ID
        dtok = AutoTokenizer.from_pretrained(MODEL_ID)
        dm = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
        from huggingface_hub import snapshot_download
        size = dir_mb(snapshot_download(MODEL_ID, allow_patterns=["*.safetensors"]))
        lat["ProtectAI DeBERTa (default, truncate 512)"] = {**timeit(lambda d: score_documents(dm, dtok, d, windowed=False, max_length=512)), "size_mb": size}
        lat["ProtectAI DeBERTa + our windowing"] = {**timeit(lambda d: score_documents(dm, dtok, d, windowed=True, max_length=512)), "size_mb": size}
    lat["quantization_accuracy_change"] = {
        k: main_tab["ToolGuard int8 (dynamic quantization)"][k]["value"] - main_tab["ToolGuard (MiniLM, windowed)"][k]["value"]
        for k in ["f1", "auroc", "recall", "recall_at_1fpr"]}
    lat["threads"] = torch.get_num_threads()
    (R / "latency.json").write_text(json.dumps(lat, indent=2))

    # ------------------------------------------------ errors
    m = models["ToolGuard (MiniLM, windowed)"]
    order_fp = [i for i in np.argsort(-m["score"]) if y[i] == 0 and m["pred"][i]][:10]
    order_fn = [i for i in np.argsort(m["score"]) if y[i] == 1 and not m["pred"][i]][:10]
    errs = {"false_positives": [], "false_negatives": []}
    for key, ids in [("false_positives", order_fp), ("false_negatives", order_fn)]:
        for i in ids:
            r = test[i]
            sp = m["span"][i]
            errs[key].append({"index": int(i), "score": float(m["score"][i]), "source": r["source"], "fmt": r["fmt"], "length": r["length"],
                              "depth": r["depth"], "attack": r.get("attack"), "top_window_excerpt": r["text"][sp[0]:sp[1]][:400]})
    (R / "errors_raw.json").write_text(json.dumps(errs, indent=2))
    np.save(R / "test_scores.npy", {k: np.asarray(v["score"]) for k, v in models.items()}, allow_pickle=True)
    charts(main_tab, dil)
    print(json.dumps({k: {m: round(v[m]["value"], 3) for m in ["precision", "recall", "f1", "auroc", "recall_at_1fpr"]} | {"fpr_hn": round(v["fpr_real_hard_negatives"]["value"], 3)} for k, v in main_tab.items()}, indent=1))


def charts(main_tab, dil):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(main_tab)
    colors = ["#8c8c8c", "#b07aa1", "#e15759", "#f28e2b", "#4e79a7", "#76b7b2", "#59a14f", "#9c755f"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, key, xs, xlabel in [(axes[0], "by_depth", [str(d) for d in DEPTHS], "Injection depth (fraction of document)"),
                                (axes[1], "by_length", [str(L) for L in LENGTHS], "Document length (approx. tokens)")]:
        for n, c in zip(names, colors):
            v = [dil[key][n][x]["value"] for x in xs]
            lo = [dil[key][n][x]["lo"] for x in xs]
            hi = [dil[key][n][x]["hi"] for x in xs]
            ax.plot(xs, v, marker="o", label=n, color=c, lw=2)
            ax.fill_between(xs, lo, hi, color=c, alpha=0.08)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall on unseen human-written attacks")
    axes[0].set_ylim(0, 1.02)
    axes[1].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Dilution: recall vs. where and how deep the injection is buried (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(R / "dilution.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    v = [main_tab[n]["fpr_real_hard_negatives"]["value"] for n in names]
    err = [[v[i] - main_tab[n]["fpr_real_hard_negatives"]["lo"] for i, n in enumerate(names)],
           [main_tab[n]["fpr_real_hard_negatives"]["hi"] - v[i] for i, n in enumerate(names)]]
    ax.barh(names, v, xerr=err, color=colors[:len(names)], capsize=3)
    for i, x in enumerate(v):
        ax.text(x + 0.01, i, f"{x:.1%}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("False-positive rate on 60 real instruction-heavy documents")
    ax.set_title("False alarms on harmless how-to and README text (95% bootstrap CI)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(R / "false_alarms.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
