"""
EXA Search API integration for NB3 multiagent system
--------------------------------------------------
Provides web search capabilities using EXA API for real-time information retrieval
when local RAG knowledge is insufficient or needs to be augmented.

Features:
- Web search with automatic result filtering
- Content extraction and summarization
- Integration with telemetry and tracing
- Graceful fallbacks when EXA is unavailable
"""

import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from opentelemetry import trace

@dataclass
class ExaResult:
    """Structured EXA search result"""
    title: str
    url: str
    text: str
    published_date: Optional[str] = None
    score: Optional[float] = None
    domain: Optional[str] = None

class ExaSearchTool:
    """EXA search tool with telemetry integration"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        self.base_url = base_url or os.getenv("EXA_BASE_URL", "https://api.exa.ai")
        self.enabled = bool(self.api_key)
        self.tracer = trace.get_tracer(__name__)

        if not self.enabled:
            print("Warning: EXA_API_KEY not found. EXA search will be disabled.")

    def search(
        self,
        query: str,
        num_results: int = 6,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        use_autoprompt: bool = True,
        type: str = "neural",
        contents: str = "text"
    ) -> List[ExaResult]:
        """
        Search using EXA API with telemetry tracking

        Args:
            query: Search query
            num_results: Number of results to return (default 6)
            include_domains: Domains to include in search
            exclude_domains: Domains to exclude from search
            start_published_date: Start date filter (YYYY-MM-DD)
            end_published_date: End date filter (YYYY-MM-DD)
            use_autoprompt: Whether to use EXA's autoprompt feature
            type: Search type ("neural" or "keyword")
            contents: Content type to return ("text" or "highlights")
        """

        with self.tracer.start_as_current_span("exa.search") as span:
            span.set_attribute("exa.query", query[:200])
            span.set_attribute("exa.num_results", num_results)
            span.set_attribute("exa.type", type)
            span.set_attribute("exa.enabled", self.enabled)

            if not self.enabled:
                span.set_attribute("exa.disabled_reason", "no_api_key")
                return []

            try:
                import requests

                headers = {
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json"
                }

                payload = {
                    "query": query,
                    "num_results": num_results,
                    "use_autoprompt": use_autoprompt,
                    "type": type,
                    "contents": {
                        "text": True if contents == "text" else False,
                        "highlights": True if contents == "highlights" else False
                    }
                }

                # Add optional filters
                if include_domains:
                    payload["include_domains"] = include_domains
                if exclude_domains:
                    payload["exclude_domains"] = exclude_domains
                if start_published_date:
                    payload["start_published_date"] = start_published_date
                if end_published_date:
                    payload["end_published_date"] = end_published_date

                span.set_attribute("exa.payload_size", len(json.dumps(payload)))

                response = requests.post(
                    f"{self.base_url}/search",
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                span.set_attribute("exa.response_status", response.status_code)
                response.raise_for_status()

                data = response.json()
                results = []

                for item in data.get("results", []):
                    result = ExaResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        text=item.get("text", ""),
                        published_date=item.get("published_date"),
                        score=item.get("score"),
                        domain=item.get("url", "").split("//")[-1].split("/")[0] if item.get("url") else None
                    )
                    results.append(result)

                span.set_attribute("exa.results_count", len(results))
                span.set_attribute("exa.total_text_length", sum(len(r.text) for r in results))

                return results

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", True)
                print(f"EXA search failed: {e}")
                return []

    def search_with_summary(
        self,
        query: str,
        num_results: int = 6,
        max_text_length: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search and return results with summary statistics
        """

        with self.tracer.start_as_current_span("exa.search_with_summary") as span:
            results = self.search(query, num_results, **kwargs)

            # Truncate text if too long
            for result in results:
                if len(result.text) > max_text_length:
                    result.text = result.text[:max_text_length] + "..."

            summary = {
                "query": query,
                "results_count": len(results),
                "results": results,
                "domains": list(set(r.domain for r in results if r.domain)),
                "total_text_length": sum(len(r.text) for r in results),
                "average_score": sum(r.score for r in results if r.score) / len(results) if results and any(r.score for r in results) else None
            }

            span.set_attribute("exa.summary.domains_count", len(summary["domains"]))
            span.set_attribute("exa.summary.avg_score", summary["average_score"] or 0)

            return summary

# Convenience functions for direct use
def exa_search(query: str, num_results: int = 6, **kwargs) -> List[ExaResult]:
    """Direct EXA search function"""
    tool = ExaSearchTool()
    return tool.search(query, num_results, **kwargs)

def exa_search_summary(query: str, num_results: int = 6, **kwargs) -> Dict[str, Any]:
    """Direct EXA search with summary"""
    tool = ExaSearchTool()
    return tool.search_with_summary(query, num_results, **kwargs)