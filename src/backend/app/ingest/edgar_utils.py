"""
EDGAR ingestion utilities — refactored to use centralized config/linking/chunkers.
- One source of truth: see app.ingest.config for endpoints & sizes.
- Professional chunking: token-based, sentence-aware, section-first.
- Consistent links: canonical doc_url + text-fragment deep links per chunk.
"""

from __future__ import annotations

import os
import re
import time
import html
import logging
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass

from openai import OpenAI

from .config import (
    SEC_SUBMISSIONS_URL_TEMPLATE, REQUEST_TIMEOUT_S, default_headers,
    EMBED_MODEL, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, VECTOR_SIZE
)
from .linker import (
    build_filing_urls, make_doc_id, make_chunk_id, normalize_cik, build_text_fragment_link
)
from .chunkers import split_by_items, pack_section_to_chunks, deep_link_snippet

# ---------- HTTP helpers ----------

def _get_json(url: str, host: str) -> dict:
    resp = requests.get(url, headers=default_headers(host), timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()

def _get_text(url: str, host: str) -> str:
    resp = requests.get(url, headers=default_headers(host), timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "").lower()
    raw = resp.text
    if "html" in ctype or raw.lstrip().lower().startswith("<html"):
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
    else:
        text = raw
    text = html.unescape(text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()

# ---------- Public API ----------

def list_filings(cik: str | int, forms: Optional[List[str]] = None, limit: int = 50, start_date: str | None = None, end_date: str | None = None) -> List[dict]:
    """List recent filings via the official submissions JSON."""
    c = normalize_cik(cik)
    url = SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=c)
    data = _get_json(url, host="data.sec.gov")
    recent = data.get("filings", {}).get("recent", {})
    keys = ["accessionNumber", "filingDate", "form", "primaryDocument", "reportDate"]
    rows = []
    n = len(recent.get("accessionNumber", []))
    for i in range(n):
        row = {k: recent.get(k, [None]*n)[i] for k in keys}
        if not row.get("accessionNumber"):
            continue
        # date filter
        if start_date and row.get("filingDate") and row["filingDate"] < start_date:
            continue
        if end_date and row.get("filingDate") and row["filingDate"] > end_date:
            continue
        if forms and row.get("form") not in set(forms):
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows

@dataclass
class Point:
    id: str
    vector: list[float]
    payload: dict

def fetch_embed_and_chunk(
    client_openai: OpenAI,
    cik: str | int,
    company: str,
    form: str,
    filing_date: str,
    accession: str,
    primary_document: str,
    embed_model: str = EMBED_MODEL,
) -> List[Point]:
    """Fetch one primary document, chunk it, embed chunks, return Qdrant-ready points."""
    urls = build_filing_urls(cik, accession, primary_document)
    doc_url = urls["doc"]
    full_text = _get_text(doc_url, host="www.sec.gov")

    doc_id = make_doc_id(cik, accession)
    sections = split_by_items(full_text)

    # Build chunks across sections
    chunks = []
    for item_id, section_title, start, end in sections:
        section_text = full_text[start:end]
        pk = pack_section_to_chunks(section_text, item_id, section_title)
        for ch in pk:
            snippet = deep_link_snippet(ch.text)
            link = build_text_fragment_link(doc_url, snippet)
            payload = {
                "doc_id": doc_id,
                "cik": normalize_cik(cik),
                "company": company,
                "form_type": form,
                "filing_date": filing_date,
                "accession": accession,
                "primary_document": primary_document,
                "item": item_id,
                "section_title": section_title,
                "chunk_idx": len(chunks),
                "char_start": ch.char_start + start,
                "char_end": ch.char_end + start,
                "est_start_tok": ch.est_start_tok,
                "est_end_tok": ch.est_end_tok,
                "chunk_tokens_budget": {"max": CHUNK_MAX_TOKENS, "overlap": CHUNK_OVERLAP_TOKENS},
                "source_url": doc_url,
                "link": link,
                "text": ch.text,
            }
            cid = make_chunk_id(doc_id, item_id, payload["chunk_idx"], extra=str(ch.est_start_tok))
            chunks.append((cid, payload))

    # Embed in batches
    texts = [p[1]["text"] for p in chunks]
    vectors = []
    B = 64
    for i in range(0, len(texts), B):
        res = client_openai.embeddings.create(model=embed_model, input=texts[i:i+B])
        vectors.extend([d.embedding for d in res.data])

    points = [Point(id=cid, vector=vectors[i], payload=pl) for i, (cid, pl) in enumerate(chunks)]
    return points

def ingest_company_filings(
    client_openai: OpenAI,
    cik: str | int,
    company: str,
    form_types: List[str],
    limit_per_form: int = 3,
    embed_model: str = EMBED_MODEL,
    sleep_between: float = 0.2,
) -> List[dict]:
    """End-to-end ingest for a company across multiple forms. Returns [{id, vector, payload}]."""
    filings = list_filings(cik, forms=form_types, limit=sum([limit_per_form]*len(form_types)))
    all_points: List[dict] = []
    for f in filings:
        try:
            pts = fetch_embed_and_chunk(
                client_openai=client_openai,
                cik=cik,
                company=company,
                form=f.get("form"),
                filing_date=f.get("filingDate"),
                accession=f.get("accessionNumber"),
                primary_document=f.get("primaryDocument"),
                embed_model=embed_model,
            )
            for p in pts:
                all_points.append({"id": p.id, "vector": p.vector, "payload": p.payload})
            time.sleep(sleep_between)
        except Exception as e:
            logging.warning("Failed filing %s: %s", f, e)
    return all_points
