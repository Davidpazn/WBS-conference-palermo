"""
Qdrant admin helpers for NB2/NB3
--------------------------------
Create/manage a Qdrant collection for EDGAR RAG, upsert points,
and provide convenience search with metadata filters and (optional) full-text match.

- Dense vectors: use OpenAI `text-embedding-3-*` (set `vector_size` accordingly).
- Payload indexes: keyword indexes on structured fields; full-text index for `text` (optional).
- Filters: support CIK, form_type, item, filing_date ranges, company exact/contains.

See also: app.ingest.edgar_utils for building point payloads.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, MatchAny, MatchText, Range, Condition
)

# ---------- Collection management ----------

def ensure_collection_edgar(
    client: QdrantClient,
    name: str,
    vector_size: int,
    on_disk: bool = True,
    recreate: bool = False,
    create_fulltext_index: bool = True,
) -> None:
    """Create (or recreate) an EDGAR collection with payload indexes."""
    try:
        if recreate:
            client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE, on_disk=on_disk),
            )
        else:
            # create if missing
            cols = client.get_collections().collections
            if name not in [c.name for c in cols]:
                client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE, on_disk=on_disk),
                )

        # Create payload indexes with error handling for duplicates
        index_fields = [
            ("cik", "keyword"),
            ("company", "keyword"),
            ("form_type", "keyword"),
            ("item", "keyword"),
            ("filing_date", "keyword"),
            ("category", "keyword"),  # For portfolio rules
            ("severity", "keyword"),  # For portfolio rules
            ("regulation", "keyword"),  # For portfolio rules
            ("source", "keyword"),  # For portfolio rules
        ]

        for field_name, field_schema in index_fields:
            try:
                client.create_payload_index(collection_name=name, field_name=field_name, field_schema=field_schema)
            except Exception:
                # Index might already exist, continue
                pass

        # Optional full-text index on main text for MatchText queries
        if create_fulltext_index:
            try:
                client.create_payload_index(
                    collection_name=name,
                    field_name="text",
                    field_schema=models.TextIndexParams(
                        type="text",
                        tokenizer=models.TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=20,
                        lowercase=True,
                        # phrase_matching=True,  # enable if running Qdrant >=1.15
                    ),
                )
            except Exception:
                # Index might already exist, continue
                pass

    except Exception as e:
        raise RuntimeError(f"Failed to create collection {name}: {e}")

# ---------- Upsert ----------

def upsert_points(client: QdrantClient, name: str, points: List[Dict[str, Any]]) -> None:
    """Upsert already-embedded points of shape: {id, vector, payload}"""
    client.upsert(
        collection_name=name,
        points=[PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points],
        wait=True,
    )

# ---------- Filter builder ----------

def build_filter(
    cik: Optional[str] = None,
    form_types: Optional[List[str]] = None,
    items: Optional[List[str]] = None,
    company_exact: Optional[str] = None,
    company_contains: Optional[str] = None,
    start_date: Optional[str] = None,   # "YYYY-MM-DD"
    end_date: Optional[str] = None,
    text_query: Optional[str] = None,   # requires full-text index
    categories: Optional[List[str]] = None,  # For portfolio rules filtering
    severity: Optional[List[str]] = None,    # For portfolio rules filtering
    regulation: Optional[str] = None,        # For portfolio rules filtering
) -> Optional[Filter]:
    must: List[Condition] = []
    if cik:
        must.append(FieldCondition(key="cik", match=MatchValue(value=cik)))
    if form_types:
        if len(form_types) == 1:
            must.append(FieldCondition(key="form_type", match=MatchValue(value=form_types[0])))
        else:
            must.append(FieldCondition(key="form_type", match=MatchAny(any=form_types)))
    if items:
        if len(items) == 1:
            must.append(FieldCondition(key="item", match=MatchValue(value=items[0])))
        else:
            must.append(FieldCondition(key="item", match=MatchAny(any=items)))
    if company_exact:
        must.append(FieldCondition(key="company", match=MatchValue(value=company_exact)))
    if company_contains:
        # substring match via MatchText or keyword contains; MatchText benefits from full-text index
        must.append(FieldCondition(key="company", match=MatchText(text=company_contains)))
    if text_query:
        must.append(FieldCondition(key="text", match=MatchText(text=text_query)))
    # Date range — if you also store an integer yyyy field, prefer that for clean range filters.
    if start_date or end_date:
        # If you store 'filing_year' as integer, switch to Range logic on that key.
        # For ISO string dates, simple lexical compare works for ranges within the same format.
        if start_date:
            must.append(FieldCondition(key="filing_date", range=Range(gte=start_date)))
        if end_date:
            must.append(FieldCondition(key="filing_date", range=Range(lte=end_date)))

    # Portfolio rules filtering
    if categories:
        if len(categories) == 1:
            must.append(FieldCondition(key="category", match=MatchValue(value=categories[0])))
        else:
            must.append(FieldCondition(key="category", match=MatchAny(any=categories)))
    if severity:
        if len(severity) == 1:
            must.append(FieldCondition(key="severity", match=MatchValue(value=severity[0])))
        else:
            must.append(FieldCondition(key="severity", match=MatchAny(any=severity)))
    if regulation:
        must.append(FieldCondition(key="regulation", match=MatchValue(value=regulation)))

    return Filter(must=must) if must else None

# ---------- Search ----------

def search_dense(
    client: QdrantClient,
    name: str,
    query_vector: List[float],
    limit: int = 10,
    query_filter: Optional[Filter] = None,
) -> List:
    """Standard dense vector search with optional structured filter / full-text conditions."""
    return client.search(
        collection_name=name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

def search_dense_by_text(
    client: QdrantClient,
    name: str,
    query_text: str,
    limit: int = 10,
    query_filter: Optional[Filter] = None,
    openai_client=None,
    embed_model: str = "text-embedding-3-small",
) -> List:
    """Text-based dense vector search with automatic embedding generation."""
    from openai import OpenAI

    # Use provided client or create new one
    if openai_client is None:
        openai_client = OpenAI()

    # Generate embedding for query text
    response = openai_client.embeddings.create(
        model=embed_model,
        input=[query_text]
    )
    query_vector = response.data[0].embedding

    # Perform vector search
    return search_dense(
        client=client,
        name=name,
        query_vector=query_vector,
        limit=limit,
        query_filter=query_filter,
    )

# ---------- Convenience: end-to-end ingest ----------

def ingest_points_batch(
    client: QdrantClient,
    name: str,
    points: List[Dict[str, Any]],
    batch_size: int = 256,
) -> None:
    for i in range(0, len(points), batch_size):
        upsert_points(client, name, points[i:i+batch_size])

# ---------- Utilities ----------

def delete_collection(client: QdrantClient, name: str) -> None:
    client.delete_collection(name)

def stat_collection(client: QdrantClient, name: str) -> dict:
    return client.get_collection(name).dict()
