"""Build the ToolGuard train/val/test sets.

Train attacks: deepset/prompt-injections + jackhhao/jailbreak-classification.
Test attacks:  Lakera/gandalf_ignore_instructions (different authors, never trained on).
Carriers:      WikiText-103 (validation split for train, test split for test) and real README
               text from permissively licensed Python packages (split by package).
Hard negatives: template-generated for train, real Dolly-15k how-to answers + real README
               install/usage sections for test.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import random
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
LENGTHS = [200, 500, 1000, 2000]  # approximate tokens, realised as words / 1.33
FORMATS = ["plain", "html", "markdown", "json"]
SEED = 13


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


# ---------------------------------------------------------------- sources
def load_train_attacks():
    from datasets import load_dataset
    rows = []
    d = load_dataset("deepset/prompt-injections")
    for split in ("train", "test"):
        for r in d[split]:
            rows.append((r["text"].strip(), int(r["label"]), "deepset"))
    j = load_dataset("jackhhao/jailbreak-classification")
    for split in ("train", "test"):
        for r in j[split]:
            t = r["prompt"].strip()
            if len(t.split()) <= 350:  # keep very long DAN-style prompts from dominating a window
                rows.append((t, int(r["type"] == "jailbreak"), "jackhhao"))
    attacks = sorted({t for t, y, _ in rows if y == 1 and t})
    benign = sorted({t for t, y, _ in rows if y == 0 and t})
    return attacks, benign


def load_test_attacks():
    from datasets import load_dataset
    d = load_dataset("Lakera/gandalf_ignore_instructions")
    return sorted({r["text"].strip() for s in ("train", "validation", "test") for r in d[s] if r["text"].strip()})


def _detok(s: str) -> str:
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([,.;:!?%)])", r"\1", s)
    s = re.sub(r"([(]) ", r"\1", s)
    return s.replace(" 's", "'s").replace(" n't", "n't").strip()


def load_wiki(split: str):
    """Return list of articles, each a list of paragraphs."""
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    arts, cur = [], []
    for r in d:
        t = r["text"]
        if re.match(r"^ = [^=].* = \n$", t):
            if len(cur) >= 2:
                arts.append(cur)
            cur = []
        elif t.strip() and not t.strip().startswith("="):
            p = _detok(t)
            if len(p.split()) >= 25:
                cur.append(p)
    if len(cur) >= 2:
        arts.append(cur)
    return arts


PERMISSIVE = re.compile(r"\bMIT\b|BSD|Apache|PSF|Python Software Foundation")


def load_readmes():
    """Real README text shipped in installed package metadata (permissive licenses only)."""
    out = {}
    for dist in md.distributions():
        meta = dist.metadata
        name = meta.get("Name") or ""
        desc = meta.get_payload() or meta.get("Description") or ""
        lic = (meta.get("License-Expression") or "") + " " + (meta.get("License") or "")[:200] + " " + " ".join(meta.get_all("Classifier") or [])
        if name and len(desc.split()) > 300 and PERMISSIVE.search(lic):
            blocks = [b.strip() for b in re.split(r"\n\s*\n", desc) if len(b.split()) >= 8]
            if len(blocks) >= 6:
                out[name.lower()] = blocks
    return dict(sorted(out.items()))


HOWTO = re.compile(r"^(how (do|can|should|to)|what are (the|some) (steps|ways|tips)|give me (steps|instructions|tips)|explain how|steps to|tips for)", re.IGNORECASE)
INSTALL = re.compile(r"\b(pip install|install|run|click|open|set|use|import|call|type|create|add)\b", re.IGNORECASE)


def load_real_hard_negatives(test_readmes):
    from datasets import load_dataset
    d = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly = [r["response"].strip() for r in d if HOWTO.search(r["instruction"].strip()) and len(r["response"].split()) >= 40]
    readme_instr = []
    for blocks in test_readmes.values():
        readme_instr += [b for b in blocks if len(INSTALL.findall(b)) >= 2]
    return sorted(set(dolly)), sorted(set(readme_instr))


# ---------------------------------------------------------------- template hard negatives (train only)
def template_hard_negative(rng: random.Random) -> str:
    pkg = rng.choice(["requests", "numpy", "flask", "pandas", "httpx", "rich", "pydantic", "uvicorn", "tqdm", "click"])
    kind = rng.randrange(5)
    if kind == 0:
        steps = [f"Run `pip install {pkg}` in your terminal.", "Create a virtual environment first with `python -m venv .venv`.",
                 f"Import it with `import {pkg}` at the top of your script.", "Restart your shell so the PATH changes take effect.",
                 "Copy `config.example.yaml` to `config.yaml` and edit the values.", "Run the test suite with `pytest -q` before you commit.",
                 "Set the `LOG_LEVEL` environment variable to `debug` to see verbose output.", "Delete the `build/` folder and run the build again."]
        return "## Installation\n\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(rng.sample(steps, rng.randint(3, 6))))
    if kind == 1:
        steps = ["Click **Settings** in the top right corner.", "Select the **Privacy** tab.", "Toggle **Sync** off and click **Save**.",
                 "Open the File menu and choose Export.", "Enter your display name and press Enter.", "Drag the file onto the upload area.",
                 "Ignore the warning banner if you are on the beta channel.", "Forget any saved Wi-Fi networks you no longer use.",
                 "Reset the form by clicking Clear.", "Close the dialog and refresh the page."]
        return "To change this setting: " + " ".join(rng.sample(steps, rng.randint(3, 6)))
    if kind == 2:
        dish = rng.choice(["banana bread", "tomato soup", "fried rice", "pancakes", "lentil curry"])
        steps = ["Preheat the oven to 180C.", "Whisk the eggs and sugar until pale.", "Chop the onions finely and fry them in oil.",
                 "Add the rice and stir for two minutes.", "Season with salt and pepper to taste.", "Let it rest for ten minutes before serving.",
                 "Ignore the timer and check the colour instead.", "Fold in the flour gently.", "Simmer on low heat for 20 minutes."]
        return f"How to make {dish}:\n" + "\n".join("- " + s for s in rng.sample(steps, rng.randint(4, 7)))
    if kind == 3:
        svc = rng.choice(["api-gateway", "billing", "search", "auth", "worker"])
        steps = [f"Page the on-call engineer for {svc}.", "Check the dashboard for error-rate spikes.", f"Run `kubectl rollout restart deploy/{svc}`.",
                 "If the restart fails, roll back to the previous release.", "Disable the feature flag and override the default timeout.",
                 "Do not delete the persistent volume.", "Post a status update in the incident channel every 30 minutes.",
                 "Once stable, write the postmortem within 48 hours."]
        return f"Runbook: {svc} high latency\n" + "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(rng.sample(steps, rng.randint(4, 7))))
    notes = ["Please reply to this email with your availability for next week.", "Remember to submit your timesheet by Friday.",
             "You must use your badge to enter the building after 6pm.", "Always lock your screen when you leave your desk.",
             "Forward this message to anyone on your team who missed the meeting.", "Do not share the meeting link outside the company.",
             "Update your password every 90 days as required by IT policy.", "Ignore the previous calendar invite, the room has changed."]
    return "Hi all,\n\n" + " ".join(rng.sample(notes, rng.randint(3, 5))) + "\n\nThanks,\nOps team"


# ---------------------------------------------------------------- carriers
def _collect(blocks_pool, rng, target_words, max_blocks=400):
    out, n = [], 0
    src = rng.choice(blocks_pool)
    i = rng.randrange(len(src))
    while n < target_words and len(out) < max_blocks:
        if i >= len(src):
            src, i = rng.choice(blocks_pool), 0
        b = src[i]
        i += 1
        out.append(b)
        n += len(b.split())
    return out


def _trim(blocks, target_words):
    # drop trailing words so docs land close to the target length
    words = sum(len(b.split()) for b in blocks)
    if words > target_words * 1.15 and len(blocks) > 1:
        excess = words - target_words
        last = blocks[-1].split()
        if len(last) > excess:
            blocks[-1] = " ".join(last[: len(last) - excess])
    return blocks


def render(fmt: str, blocks: list[str], insert: str | None, depth: float, rng: random.Random):
    """Render carrier blocks in ``fmt`` with ``insert`` placed at relative ``depth``.
    Returns (document, (char_start, char_end) of the inserted text or None)."""
    k = round(depth * len(blocks))
    marker = "\x00INS\x00"
    items = list(blocks)
    if insert is not None:
        items = items[:k] + [marker] + items[k:]
    if fmt == "plain":
        doc = "\n\n".join(items)
    elif fmt == "html":
        title = rng.choice(["Article", "Overview", "Encyclopedia entry", "Reference"])
        parts = [f"<!DOCTYPE html>\n<html><head><title>{title}</title></head>\n<body>\n<nav><a href=\"/\">Home</a> | <a href=\"/about\">About</a></nav>\n<main>"]
        for it in items:
            parts.append("<!-- " + it + " -->" if it == marker else "<p>" + it + "</p>")
        parts.append("</main>\n<footer>Content available under CC BY-SA.</footer>\n</body></html>")
        doc = "\n".join(parts)
    elif fmt == "markdown":
        doc = "\n\n".join(items)
    elif fmt == "json":
        results = []
        for i, it in enumerate(items):
            results.append({"id": i, "title": f"Result {i + 1}", "snippet": it, "score": round(rng.random(), 3)})
        doc = json.dumps({"status": "ok", "query": "user search", "results": results}, ensure_ascii=False, indent=1)
    else:
        raise ValueError(fmt)
    if insert is None:
        return doc, None
    enc_insert = json.dumps(insert, ensure_ascii=False)[1:-1] if fmt == "json" else insert
    enc_marker = json.dumps(marker)[1:-1] if fmt == "json" else marker
    start = doc.index(enc_marker)
    doc = doc.replace(enc_marker, enc_insert)
    return doc, (start, start + len(enc_insert))


def make_doc(rng, fmt, wiki, readmes, target_tokens, insert=None, depth=0.5, filler=None):
    words = int(target_tokens / 1.33)
    ins_words = len(insert.split()) if insert else 0
    pool = filler if filler is not None else (readmes if fmt == "markdown" else wiki)
    blocks = _trim(_collect(pool, rng, max(20, words - ins_words)), max(20, words - ins_words))
    return render(fmt, blocks, insert, depth, rng)


# ---------------------------------------------------------------- leakage
def shingles(t: str, n: int = 5):
    t = re.sub(r"\s+", " ", t.lower()).strip()
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def max_jaccard(queries, refs, n=5):
    ref_sh = [shingles(r, n) for r in refs]
    # inverted index over shingles to keep this fast
    inv = {}
    for j, s in enumerate(ref_sh):
        for g in s:
            inv.setdefault(g, []).append(j)
    out = []
    for q in queries:
        qs = shingles(q, n)
        counts = {}
        for g in qs:
            for j in inv.get(g, ()):
                counts[j] = counts.get(j, 0) + 1
        best = 0.0
        for j, c in counts.items():
            best = max(best, c / (len(qs) + len(ref_sh[j]) - c))
        out.append(best)
    return out


# ---------------------------------------------------------------- build
def build(n_train_docs=6000, n_val_docs=200, per_cell=6, n_neg=60, seed=SEED, jaccard_threshold=0.5):
    """Defaults were cut from (800 val, 30 per cell, 300+300 negatives) to fit a swap-bound 8 GB CPU; see README."""
    rng = random.Random(seed)
    DATA.mkdir(exist_ok=True)
    tr_attacks, tr_benign = load_train_attacks()
    te_attacks_all = load_test_attacks()
    wiki_tr, wiki_te = load_wiki("validation"), load_wiki("test")
    readmes = load_readmes()
    rd_tr = {k: v for k, v in readmes.items() if _h(k) % 3 != 0}
    rd_te = {k: v for k, v in readmes.items() if _h(k) % 3 == 0}
    dolly_hn, readme_hn = load_real_hard_negatives(rd_te)

    # ---- leakage audit on attacks
    sims = max_jaccard(te_attacks_all, tr_attacks + tr_benign)
    te_attacks = [a for a, s in zip(te_attacks_all, sims) if s < jaccard_threshold]
    exact = len(set(te_attacks_all) & set(tr_attacks + tr_benign))
    hn_sims = max_jaccard(dolly_hn, tr_benign)
    dolly_hn = [t for t, s in zip(dolly_hn, hn_sims) if s < jaccard_threshold]
    wiki_tr_set = {p for a in wiki_tr for p in a}
    wiki_overlap = sum(p in wiki_tr_set for a in wiki_te for p in a)
    audit = {
        "method": f"character 5-gram Jaccard, remove test item if max similarity to any train text >= {jaccard_threshold}",
        "test_attacks_before": len(te_attacks_all), "test_attacks_removed": len(te_attacks_all) - len(te_attacks),
        "test_attacks_exact_duplicates_of_train": exact, "test_attacks_after": len(te_attacks),
        "test_attack_similarity_quantiles": {q: float(sorted(sims)[int(q * (len(sims) - 1))]) for q in (0.5, 0.9, 0.99)},
        "test_hard_negatives_removed": int(sum(s >= jaccard_threshold for s in hn_sims)),
        "wiki_test_paragraphs_found_in_train_carriers": wiki_overlap,
        "readme_packages_train": sorted(rd_tr), "readme_packages_test": sorted(rd_te),
    }

    # ---- train/val split of train-source texts
    rng.shuffle(tr_attacks)
    rng.shuffle(tr_benign)
    nva, nvb = len(tr_attacks) // 10, len(tr_benign) // 10
    split_att = {"val": tr_attacks[:nva], "train": tr_attacks[nva:]}
    split_ben = {"val": tr_benign[:nvb], "train": tr_benign[nvb:]}
    rng.shuffle(wiki_tr)
    nw = len(wiki_tr) // 10
    split_wiki = {"val": wiki_tr[:nw], "train": wiki_tr[nw:]}
    rk = sorted(rd_tr)
    split_rd = {"val": [rd_tr[k] for k in rk[:3]], "train": [rd_tr[k] for k in rk[3:]]}

    def train_like(split, n, use_hard_neg=True):
        r = random.Random(seed + (1 if split == "val" else 0))
        rows = []
        for i in range(n):
            u = r.random()
            fmt = r.choice(FORMATS)
            L = r.choice(LENGTHS)
            dep = r.choice(DEPTHS)
            base = dict(split=split, fmt=fmt, length=L, depth=None, source=None, span=None)
            if u < 0.40:  # injection in carrier
                a = r.choice(split_att[split])
                doc, span = make_doc(r, fmt, split_wiki[split], split_rd[split], L, a, dep)
                rows.append({**base, "text": doc, "label": 1, "depth": dep, "span": span, "source": "train_attack_in_carrier"})
            elif u < 0.48:  # raw short attack
                a = r.choice(split_att[split])
                rows.append({**base, "text": a, "label": 1, "fmt": "raw", "length": 0, "span": (0, len(a)), "source": "train_attack_raw"})
            elif u < 0.68:  # clean carrier
                doc, _ = make_doc(r, fmt, split_wiki[split], split_rd[split], L)
                rows.append({**base, "text": doc, "label": 0, "source": "carrier_clean"})
            elif u < 0.76:  # raw benign prompt
                b = r.choice(split_ben[split])
                rows.append({**base, "text": b, "label": 0, "fmt": "raw", "length": 0, "source": "benign_prompt"})
            else:  # template hard negative inside carrier (or plain clean carrier when disabled)
                if use_hard_neg:
                    hn = template_hard_negative(r)
                    doc, span = make_doc(r, fmt, split_wiki[split], split_rd[split], L, hn, dep)
                    rows.append({**base, "text": doc, "label": 0, "depth": dep, "span": span, "source": "template_hard_negative"})
                else:
                    doc, _ = make_doc(r, fmt, split_wiki[split], split_rd[split], L)
                    rows.append({**base, "text": doc, "label": 0, "source": "carrier_clean"})
        return rows

    train = train_like("train", n_train_docs)
    train_nohn = train_like("train", n_train_docs, use_hard_neg=False)
    val = train_like("val", n_val_docs)

    # ---- test set
    r = random.Random(seed + 100)
    test = []
    rd_te_list = [rd_te[k] for k in sorted(rd_te)]
    atk = list(te_attacks)
    r.shuffle(atk)
    ai = 0
    for dep in DEPTHS:  # 5 depths x 4 lengths x per_cell positives
        for L in LENGTHS:
            for _ in range(per_cell):
                a = atk[ai % len(atk)]
                ai += 1
                fmt = r.choice(FORMATS)
                doc, span = make_doc(r, fmt, wiki_te, rd_te_list, L, a, dep)
                test.append(dict(split="test", text=doc, label=1, fmt=fmt, length=L, depth=dep, span=span, source="gandalf_in_carrier", attack=a))
    for i in range(n_neg):  # clean real carriers
        fmt, L = FORMATS[i % 4], LENGTHS[(i // 4) % 4]
        doc, _ = make_doc(r, fmt, wiki_te, rd_te_list, L)
        test.append(dict(split="test", text=doc, label=0, fmt=fmt, length=L, depth=None, span=None, source="carrier_clean"))
    hn_blocks = [[t] for t in dolly_hn] + [[t] for t in readme_hn]
    for i in range(n_neg):  # real hard negatives: documents made only of real instruction-heavy text
        fmt, L = FORMATS[i % 4], LENGTHS[(i // 4) % 4]
        doc, _ = make_doc(r, fmt, wiki_te, rd_te_list, L, filler=hn_blocks)
        test.append(dict(split="test", text=doc, label=0, fmt=fmt, length=L, depth=None, span=None, source="real_hard_negative"))

    for name, rows in [("train", train), ("train_nohn", train_nohn), ("val", val), ("test", test)]:
        with open(DATA / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    audit["sizes"] = {k: len(v) for k, v in [("train", train), ("train_nohn", train_nohn), ("val", val), ("test", test)]}
    audit["source_counts"] = {"train_attack_texts": len(split_att["train"]), "val_attack_texts": len(split_att["val"]),
                              "test_attack_texts_used": min(len(te_attacks), per_cell * 20), "dolly_howto_hard_negative_texts": len(dolly_hn),
                              "readme_instruction_blocks_test": len(readme_hn), "wiki_train_articles": len(wiki_tr), "wiki_test_articles": len(wiki_te)}
    Path("results").mkdir(exist_ok=True)
    Path("results/leakage_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def load(name):
    with open(DATA / f"{name}.jsonl") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    a = build()
    print(json.dumps({k: v for k, v in a.items() if "packages" not in k}, indent=2))
