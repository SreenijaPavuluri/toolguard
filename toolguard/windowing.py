"""Split long documents into overlapping token windows and map them back to character spans."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Window:
    input_ids: list[int]
    char_start: int
    char_end: int


def _wrap(tokenizer, ids):
    return [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]


def make_windows(text: str, tokenizer, max_length: int = 256, stride: int = 64) -> list[Window]:
    """Tokenize ``text`` once and cut it into windows of ``max_length`` tokens (special tokens
    included) that overlap by ``stride`` tokens. Each window remembers the character span it covers."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    body = max_length - 2
    if not ids:
        return [Window(_wrap(tokenizer, []), 0, 0)]
    step = max(1, body - stride)
    windows = []
    start = 0
    while True:
        end = min(start + body, len(ids))
        wid = _wrap(tokenizer, ids[start:end])
        windows.append(Window(wid, offsets[start][0], offsets[end - 1][1]))
        if end >= len(ids):
            break
        start += step
    return windows


def truncate_window(text: str, tokenizer, max_length: int = 256) -> Window:
    """The no-windowing condition: keep only the first ``max_length`` tokens."""
    return make_windows(text, tokenizer, max_length=max_length, stride=0)[0]
