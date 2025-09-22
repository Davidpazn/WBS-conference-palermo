"""
Company index utilities: resolve CIKs from tickers or names using SEC's mapping.
- Pulls https://www.sec.gov/files/company_tickers.json and caches locally.
- Provides ticker->CIK and fuzzy name->CIK lookup.
"""

from __future__ import annotations
import os, time, json, re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import requests
from difflib import SequenceMatcher

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CACHE_DIR = Path(os.getenv("EDGAR_CACHE_DIR", ".cache/edgar"))
CACHE_PATH = CACHE_DIR / "company_tickers.json"
CACHE_TTL_SECONDS = int(os.getenv("EDGAR_CACHE_TTL", "86400"))  # 1 day

def _headers():
    ua = os.getenv("SEC_CONTACT_EMAIL", "dev@localhost")
    return {"User-Agent": f"NB-RAG-EDGAR/0.2 (+contact: {ua})"}

def _load_remote() -> Dict[str, Any]:
    resp = requests.get(SEC_TICKERS_URL, headers=_headers(), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")
    return data

def load_company_tickers() -> Dict[str, Any]:
    try:
        if CACHE_PATH.exists():
            age = time.time() - CACHE_PATH.stat().st_mtime
            if age < CACHE_TTL_SECONDS:
                return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return _load_remote()
    except Exception:
        # last-resort: stale cache
        if CACHE_PATH.exists():
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        raise

def _normalize_ticker(t: str) -> str:
    return re.sub(r"[^A-Z0-9.-]", "", t.upper())

def find_cik_by_ticker(ticker: str) -> Optional[str]:
    data = load_company_tickers()
    # SEC structure is dict with keys "0","1",... each value has 'ticker','title','cik_str'
    for _, row in data.items():
        if _normalize_ticker(row.get("ticker","")) == _normalize_ticker(ticker):
            return str(row.get("cik_str")).zfill(10)
    return None

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(a=a.lower(), b=b.lower()).ratio()

def find_cik_by_name(name: str, min_ratio: float = 0.7) -> Optional[Tuple[str, str]]:
    data = load_company_tickers()
    best = (None, 0.0, None)  # cik, score, title
    for _, row in data.items():
        title = row.get("title","")
        r = _ratio(name, title)
        if r > best[1]:
            best = (str(row.get("cik_str")).zfill(10), r, title)
    if best[0] and best[1] >= min_ratio:
        return (best[0], best[2])
    return None
