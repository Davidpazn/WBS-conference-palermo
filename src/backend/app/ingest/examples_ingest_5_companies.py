"""
Example driver: ingest 5 companies' filings from the past N years into Qdrant.
- Resolves CIKs via SEC company_tickers.json (ticker or fuzzy name).
- Filters filings by date window.
- Uses EDGAR ingest utilities for chunking/embeddings.
- Upserts into Qdrant with payload indexes.
Run:
    python -m app.ingest.examples_ingest_5_companies --tickers AAPL MSFT AMZN GOOGL META --years 5 --forms 10-K 10-Q
"""
from __future__ import annotations
import argparse, os, sys, datetime as dt
from typing import List

from openai import OpenAI
from qdrant_client import QdrantClient

from app.ingest.company_index import find_cik_by_ticker, find_cik_by_name
from app.ingest.config import EMBED_MODEL, VECTOR_SIZE, QDRANT_COLLECTION
from app.ingest.edgar_utils import ingest_company_filings, list_filings
from app.tools.qdrant_admin import ensure_collection_edgar, ingest_points_batch

def resolve_cik(identifier: str) -> str:
    """Accepts ticker or name; returns 10-digit CIK or raises."""
    cik = find_cik_by_ticker(identifier)
    if cik:
        return cik
    res = find_cik_by_name(identifier)
    if res:
        return res[0]
    raise ValueError(f"Could not resolve CIK for '{identifier}'")

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["AAPL","MSFT","AMZN","GOOGL","META"])
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--forms", nargs="+", default=["10-K","10-Q"])
    p.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION))
    args = p.parse_args(argv)

    cutoff = (dt.date.today() - dt.timedelta(days=365*args.years)).isoformat()

    oa = OpenAI()  # requires OPENAI_API_KEY
    qc = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

    ensure_collection_edgar(qc, args.collection, vector_size=VECTOR_SIZE, recreate=False)

    for ident in args.tickers:
        cik = resolve_cik(ident)
        # quick discovery pass: list filings & filter by cutoff date
        filings = list_filings(cik, forms=args.forms, limit=200, start_date=cutoff)
        if not filings:
            print(f"[WARN] No filings found since {cutoff} for {ident} ({cik})")
            continue
        company = ident  # you can replace by official title during ingest
        # ingest each filing (chunk+embed) and upsert
        for f in filings:
            points = ingest_company_filings(
                client_openai=oa, cik=cik, company=company,
                form_types=[f["form"]], limit_per_form=1,  # process this one filing
            )
            ingest_points_batch(qc, args.collection, points, batch_size=200)
            print(f"[OK] Upserted {len(points)} chunks for {ident} {f['form']} {f['filingDate']}")

if __name__ == "__main__":
    sys.exit(main())
