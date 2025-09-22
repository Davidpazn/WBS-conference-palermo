"""
Professional token-based chunking for EDGAR text.
- Splits by ITEM sections first, then by sentences.
- Packs sentences into chunks using a token budget (tiktoken if available).
- Provides start/end token estimates and char spans for better provenance.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    import tiktoken
except Exception:
    tiktoken = None

from .config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, MIN_CHUNK_TOKENS, EMBED_MODEL, DEEP_LINK_SNIPPET_TOKENS

ITEM_REGEX = re.compile(
    r'(^|\n)\s*ITEM\s+(?P<num>\d+[A-Z]?)\.?\s*(?P<title>[A-Z0-9&(),\-\/ ]{3,})\s*\n',
    re.IGNORECASE
)

@dataclass
class PackedChunk:
    text: str
    char_start: int
    char_end: int
    est_start_tok: int
    est_end_tok: int
    section_item: str
    section_title: str

def _get_tokenizer():
    if tiktoken:
        try:
            return tiktoken.encoding_for_model("gpt-4o-mini")  # robust default
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    return None

_enc = _get_tokenizer()

def _tok_len(s: str) -> int:
    if _enc:
        return len(_enc.encode(s))
    # heuristic: ~4 chars/token
    return max(1, len(s) // 4)

def _sent_split(text: str) -> List[Tuple[str, int, int]]:
    """Naive but robust sentence splitter that respects SEC formatting (newlines & headings).
    Returns list of (sentence, start, end) char spans.
    """
    # Keep paragraph/newline blocks; split on punctuation + following space/newline
    spans: List[Tuple[str, int, int]] = []
    start = 0
    buf = []
    for i, ch in enumerate(text):
        buf.append(ch)
        if ch in ".!?" and (i+1 == len(text) or text[i+1] in " \n\t\r"):
            seg = "".join(buf)
            spans.append((seg, start, i+1))
            start = i+1
            buf = []
    if buf:
        seg = "".join(buf)
        spans.append((seg, start, len(text)))
    # Merge tiny sentences
    merged: List[Tuple[str, int, int]] = []
    cur = None
    for s, a, b in spans:
        if not cur:
            cur = [s, a, b]
        else:
            if _tok_len(cur[0]) < 8:
                cur[0] += s
                cur[2] = b
            else:
                merged.append((cur[0], cur[1], cur[2]))
                cur = [s, a, b]
    if cur:
        merged.append((cur[0], cur[1], cur[2]))
    return merged

def split_by_items(full_text: str) -> List[Tuple[str, str, int, int]]:
    """Return list of (item_id, section_title, start, end) spans over full_text."""
    matches = list(ITEM_REGEX.finditer(full_text))
    out: List[Tuple[str, str, int, int]] = []
    if not matches:
        return [("ALL", "All Content", 0, len(full_text))]
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        item = (m.group("num") or "").upper()
        title = (m.group("title") or "").title()
        out.append((item, f"Item {item}. {title}", start, end))
    return out

def pack_section_to_chunks(section_text: str, section_item: str, section_title: str) -> List[PackedChunk]:
    """Pack sentences into token-bounded chunks with overlap."""
    sents = _sent_split(section_text)
    chunks: List[PackedChunk] = []
    cur_text = ""
    cur_start_char = 0
    cur_start_tok = 0
    cur_tok = 0

    i = 0
    while i < len(sents):
        s, a, b = sents[i]
        stoks = _tok_len(s)
        if cur_text == "":
            cur_text = s
            cur_start_char = a
            cur_start_tok = cur_tok
            cur_tok += stoks
        elif cur_tok + stoks <= CHUNK_MAX_TOKENS:
            cur_text += s
            cur_tok += stoks
        else:
            # emit current if it meets minimum
            if cur_tok >= MIN_CHUNK_TOKENS:
                chunks.append(PackedChunk(
                    text=cur_text.strip(),
                    char_start=cur_start_char,
                    char_end=a,  # end before current sentence
                    est_start_tok=cur_start_tok,
                    est_end_tok=cur_start_tok + cur_tok,
                    section_item=section_item,
                    section_title=section_title,
                ))
                # start overlap window
                # backtrack sentences until overlap is satisfied
                overlap = CHUNK_OVERLAP_TOKENS
                j = i - 1
                back_text = ""
                back_tokens = 0
                back_start = a
                while j >= 0 and back_tokens < overlap:
                    sj, aj, bj = sents[j]
                    back_text = sj + back_text
                    back_tokens += _tok_len(sj)
                    back_start = aj
                    j -= 1
                cur_text = back_text + s
                cur_start_char = back_start
                cur_start_tok = max(0, cur_tok - back_tokens)
                cur_tok = _tok_len(cur_text)
            else:
                # forced add to avoid tiny chunk
                cur_text += s
                cur_tok += stoks
        i += 1

    if cur_text:
        chunks.append(PackedChunk(
            text=cur_text.strip(),
            char_start=cur_start_char,
            char_end=sents[-1][2] if sents else len(section_text),
            est_start_tok=cur_start_tok,
            est_end_tok=cur_start_tok + cur_tok,
            section_item=section_item,
            section_title=section_title,
        ))
    return chunks

def deep_link_snippet(text: str) -> str:
    """Return a compact snippet for deep link text fragments."""
    # take first N tokens approximately
    words = text.strip().split()
    # approximate N by DEEP_LINK_SNIPPET_TOKENS words (not tokens), compact enough
    return " ".join(words[: DEEP_LINK_SNIPPET_TOKENS])
