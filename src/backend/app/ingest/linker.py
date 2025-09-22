"""
URL/ID normalization and deep-link helpers for EDGAR.
"""

from __future__ import annotations
import re
import hashlib
import urllib.parse as _url

from .config import SEC_ARCHIVES_PREFIX_TEMPLATE, USE_TEXT_FRAGMENT_ANCHORS, DEEP_LINK_SNIPPET_TOKENS

def normalize_cik(cik: str | int) -> str:
    s = re.sub(r"\D", "", str(cik).strip())
    return s.zfill(10)

def accession_no_nodash(accession: str) -> str:
    return re.sub(r"[^0-9]", "", accession)

def build_filing_urls(cik: str | int, accession: str, primary_document: str) -> dict:
    """Return { base, doc } URLs for the filing's primary document."""
    c_for_path = str(int(str(cik)))  # drop leading zeros in SEC path
    acc_no = accession_no_nodash(accession)
    base = SEC_ARCHIVES_PREFIX_TEMPLATE.format(cik=c_for_path, acc_no_nodash=acc_no)
    return {"base": base, "doc": base + primary_document}

def make_doc_id(cik: str | int, accession: str) -> str:
    return f"{normalize_cik(cik)}_{accession_no_nodash(accession)}"

def make_chunk_id(doc_id: str, item: str, chunk_idx: int, extra: str = "") -> str:
    h = hashlib.sha1(f"{doc_id}|{item}|{chunk_idx}|{extra}".encode()).hexdigest()
    return h

def build_text_fragment_link(doc_url: str, snippet: str) -> str:
    """Create a deep link using text fragments (Chrome/Edge/Firefox supported).
    Falls back to the plain doc_url if disabled.
    """
    if not USE_TEXT_FRAGMENT_ANCHORS or not snippet:
        return doc_url
    # Encode a compact snippet to avoid overly long URLs
    s = snippet.strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    # URL-encode for text fragment directive
    encoded = _url.quote(s, safe="")
    return f"{doc_url}#:~:text={encoded}"
