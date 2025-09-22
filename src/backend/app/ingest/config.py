"""
Central configuration & constants for EDGAR ingestion.
This is the single source of truth for endpoints, headers, chunking sizes, etc.
"""

from __future__ import annotations
import os
from dataclasses import dataclass

# ---- SEC endpoints ----
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_PREFIX_TEMPLATE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_nodash}/"

# ---- User agent / network ----
DEFAULT_CONTACT = os.getenv("SEC_CONTACT_EMAIL", "dev@localhost")
USER_AGENT = f"NB-RAG-EDGAR/0.2 (+contact: {DEFAULT_CONTACT})"
REQUEST_TIMEOUT_S = 60
RATE_LIMIT_RPS = 5  # keep polite; SEC fair-use suggests <= 10 rps

# ---- Embeddings / Qdrant ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")   # 1536 dims
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1536"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "edgar_nb")

# ---- Chunking ----
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "800"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))
MIN_CHUNK_TOKENS = int(os.getenv("MIN_CHUNK_TOKENS", "40"))

# ---- Linking ----
USE_TEXT_FRAGMENT_ANCHORS = os.getenv("USE_TEXT_FRAGMENT_ANCHORS", "true").lower() == "true"
DEEP_LINK_SNIPPET_TOKENS = int(os.getenv("DEEP_LINK_SNIPPET_TOKENS", "20"))  # how much to include in #:~:text=

def default_headers(host: str = "www.sec.gov") -> dict:
    return {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": host,
        "Connection": "keep-alive",
    }
