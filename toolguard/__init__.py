"""ToolGuard: prompt-injection scanning for agent tool outputs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Guard", "ScanResult"]
__version__ = "0.1.0"

DEFAULT_MODEL = os.environ.get("TOOLGUARD_MODEL", str(Path(__file__).resolve().parent.parent / "models" / "toolguard-minilm"))


def _quantize(model, *args, **kwargs):
    """Dynamic int8 quantization; picks an available engine (qnnpack on ARM, fbgemm on x86)."""
    import torch
    engines = torch.backends.quantized.supported_engines
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in engines else "fbgemm"
    return torch.ao.quantization.quantize_dynamic(model, *args, **kwargs)


@dataclass
class ScanResult:
    is_injection: bool
    score: float
    span: tuple[int, int] | None
    threshold: float


class Guard:
    def __init__(self, model, tokenizer, config):
        self.model, self.tokenizer, self.config = model, tokenizer, config

    @classmethod
    def load(cls, path: str | None = None, quantize: bool = False) -> Guard:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = path or DEFAULT_MODEL
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        if quantize:
            model = _quantize(model, {torch.nn.Linear}, dtype=torch.qint8)
        cfg = json.loads((Path(path) / "toolguard_config.json").read_text())
        return cls(model, tok, cfg)

    def scan(self, text: str) -> ScanResult:
        import math

        from .model import score_documents
        z, spans = score_documents(self.model, self.tokenizer, [text], windowed=True,
                                   max_length=self.config["max_length"], stride=self.config["stride"])
        p = 1 / (1 + math.exp(-max(-50.0, min(50.0, float(z[0]) / self.config["temperature"]))))
        thr = self.config["threshold"]
        return ScanResult(is_injection=p >= thr, score=p, span=tuple(spans[0]) if spans[0] else None, threshold=thr)
