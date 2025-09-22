"""
Accuracy-driven LangGraph patterns for NB3
------------------------------------------
Implements sophisticated multi-query retrieval, fusion, reranking, and verification
workflows with Human-in-the-Loop gates and compliance checking.

Features:
- Multi-query expansion for comprehensive retrieval
- Retrieval fusion with MMR deduplication
- LLM-based reranking for relevance
- Answer verification with confidence scoring
- Threshold-based escalation with EXA fallback
- Compliance and budget gates
- Full telemetry integration
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from opentelemetry import trace

from ..state import AppState
from ..tools.qdrant_admin import search_dense_by_text, build_filter
from ..tools.exa_search import exa_search_summary
from ..memory import save, recall

@dataclass
class AccuracyConfig:
    """Configuration for accuracy-driven workflow"""
    accuracy_target: float = 0.80
    max_loops: int = 2
    top_k: int = 6
    fusion_k: int = 10
    rerank_k: int = 5
    exa_num_results: int = 6
    max_tokens: int = 120000
    max_tool_calls: int = 20
    max_seconds: int = 120
    diagram_engine: str = "mermaid"
    diagram_format: str = "svg"

@dataclass
class QueryVariation:
    """A query variation with metadata"""
    query: str
    type: str  # "original", "paraphrase", "sub_question", "broader", "narrower"
    reasoning: str

@dataclass
class EvidenceItem:
    """Evidence item with scoring and metadata"""
    text: str
    source: str
    score: float
    metadata: Dict[str, Any]
    rank: Optional[int] = None

@dataclass
class VerificationResult:
    """Verification result with detailed scoring"""
    coverage: float  # How well evidence covers the question
    faithfulness: float  # How faithful answer is to evidence
    confidence: float  # Overall confidence score
    missing_aspects: List[str]
    potential_issues: List[str]
    reasoning: str

class AccuracyDrivenWorkflow:
    """Accuracy-driven workflow with all pattern implementations"""

    def __init__(self, config: Optional[AccuracyConfig] = None):
        self.config = config or AccuracyConfig()
        self.tracer = trace.get_tracer(__name__)

    def multi_query_expand_node(self, state: AppState) -> AppState:
        """Generate multiple query variations for comprehensive retrieval"""
        with self.tracer.start_as_current_span("rag.multi_query") as span:
            span.set_attribute("query.original", state.user_query[:200])
            span.set_attribute("config.fusion_k", self.config.fusion_k)

            try:
                from openai import OpenAI
                client = OpenAI()
                model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

                system_prompt = """You are a query expansion expert. Given a user query, generate diverse query variations that will help retrieve comprehensive information.

Generate 4-5 variations including:
1. Paraphrases with different wording
2. More specific sub-questions
3. Broader related questions
4. Technical/formal variations
5. Practical/application-focused variations

Return JSON array with this structure:
[
    {"query": "variation text", "type": "paraphrase", "reasoning": "why this helps"},
    {"query": "variation text", "type": "sub_question", "reasoning": "why this helps"},
    ...
]

Types: "paraphrase", "sub_question", "broader", "narrower", "technical", "practical"
Keep queries clear and searchable."""

                user_prompt = f"Original query: {state.user_query}\n\nGenerate query variations that will help find comprehensive information about this topic."

                with self.tracer.start_as_current_span("gen_ai.query_expansion") as gen_span:
                    gen_span.set_attribute("gen_ai.system", "openai")
                    gen_span.set_attribute("gen_ai.request.model", model)

                    response = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )

                    try:
                        response_text = response.output[0].content[0].text
                    except:
                        response_text = str(response)

                    # Parse variations
                    try:
                        variations_data = json.loads(response_text.strip())
                        variations = [QueryVariation(**item) for item in variations_data]
                    except:
                        # Fallback: create basic variations
                        variations = [
                            QueryVariation(state.user_query, "original", "original user query"),
                            QueryVariation(f"What is {state.user_query}?", "paraphrase", "question form"),
                            QueryVariation(f"How to {state.user_query}", "practical", "practical application"),
                        ]

                    # Always include original query
                    original_query = QueryVariation(state.user_query, "original", "original user query")
                    if not any(v.query == state.user_query for v in variations):
                        variations.insert(0, original_query)

                    state.query_variations = [asdict(v) for v in variations]
                    span.set_attribute("query.variations_count", len(variations))

            except Exception as e:
                span.record_exception(e)
                # Fallback to original query only
                state.query_variations = [{"query": state.user_query, "type": "original", "reasoning": "fallback"}]

        return state

    def retrieval_fusion_node(self, state: AppState) -> AppState:
        """Perform retrieval for each query variation and fuse results"""
        with self.tracer.start_as_current_span("rag.fusion") as span:
            span.set_attribute("fusion.input_queries", len(state.query_variations or []))

            try:
                from qdrant_client import QdrantClient

                # Initialize Qdrant client
                qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
                qdrant_api_key = os.getenv("QDRANT_API_KEY")
                collection_name = os.getenv("QDRANT_COLLECTION_NAME", "agents_knowledge")

                client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    check_compatibility=False  # Skip version check to avoid warning
                )

                all_evidence = []
                query_results = {}

                # Search for each query variation
                for variation in (state.query_variations or []):
                    query = variation["query"]

                    with self.tracer.start_as_current_span("rag.search_variation") as search_span:
                        search_span.set_attribute("query", query[:200])
                        search_span.set_attribute("variation_type", variation["type"])

                        try:
                            # Search Qdrant
                            results = search_dense_by_text(
                                client=client,
                                name=collection_name,
                                query_text=query,
                                limit=self.config.top_k,
                                query_filter=build_filter()  # Can add filters based on state
                            )

                            # Convert to evidence items
                            evidence_items = []
                            for i, result in enumerate(results):
                                evidence = EvidenceItem(
                                    text=result.payload.get("text", ""),
                                    source=result.payload.get("source", "unknown"),
                                    score=result.score,
                                    metadata={
                                        "id": result.id,
                                        "payload": result.payload,
                                        "query_variation": variation["type"],
                                        "rank_in_query": i
                                    }
                                )
                                evidence_items.append(evidence)

                            query_results[query] = evidence_items
                            all_evidence.extend(evidence_items)

                            search_span.set_attribute("results_count", len(evidence_items))

                        except Exception as e:
                            search_span.record_exception(e)
                            query_results[query] = []

                # Deduplicate using MMR-style approach (simple text similarity)
                deduplicated_evidence = self._mmr_deduplicate(all_evidence, self.config.fusion_k)

                state.fusion_evidence = [asdict(e) for e in deduplicated_evidence]
                state.query_results = {k: [asdict(e) for e in v] for k, v in query_results.items()}

                span.set_attribute("fusion.total_evidence", len(all_evidence))
                span.set_attribute("fusion.deduplicated_evidence", len(deduplicated_evidence))

            except Exception as e:
                span.record_exception(e)
                state.fusion_evidence = []
                state.query_results = {}

        return state

    def rerank_llm_node(self, state: AppState) -> AppState:
        """Rerank evidence using LLM for relevance to original query"""
        with self.tracer.start_as_current_span("rag.rerank") as span:
            span.set_attribute("rerank.input_count", len(state.fusion_evidence or []))

            try:
                from openai import OpenAI
                client = OpenAI()
                model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

                if not state.fusion_evidence:
                    state.reranked_evidence = []
                    return state

                # Prepare evidence for reranking
                evidence_texts = []
                for i, evidence in enumerate(state.fusion_evidence):
                    evidence_texts.append(f"[{i}] {evidence['text'][:500]}...")

                system_prompt = """You are a relevance ranking expert. Given a query and a list of evidence items, rank them by relevance to the query.

Return a JSON array of indices in descending order of relevance (most relevant first).
Consider:
- Direct relevance to the query
- Quality and specificity of information
- Uniqueness of information

Example: [2, 0, 4, 1, 3] means item 2 is most relevant, then item 0, etc."""

                user_prompt = f"""Query: {state.user_query}

Evidence items:
{chr(10).join(evidence_texts)}

Rank by relevance (return JSON array of indices):"""

                with self.tracer.start_as_current_span("gen_ai.rerank") as gen_span:
                    gen_span.set_attribute("gen_ai.system", "openai")
                    gen_span.set_attribute("gen_ai.request.model", model)

                    response = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )

                    try:
                        response_text = response.output[0].content[0].text
                    except:
                        response_text = str(response)

                    # Parse ranking
                    try:
                        ranking_indices = json.loads(response_text.strip())
                        if not isinstance(ranking_indices, list):
                            raise ValueError("Ranking must be a list")
                    except:
                        # Fallback: keep original order
                        ranking_indices = list(range(len(state.fusion_evidence)))

                    # Apply ranking and keep top RERANK_K
                    reranked_evidence = []
                    for rank, idx in enumerate(ranking_indices[:self.config.rerank_k]):
                        if 0 <= idx < len(state.fusion_evidence):
                            evidence = state.fusion_evidence[idx].copy()
                            evidence["rank"] = rank
                            evidence["rerank_score"] = 1.0 - (rank / len(ranking_indices))
                            reranked_evidence.append(evidence)

                    state.reranked_evidence = reranked_evidence
                    span.set_attribute("rerank.output_count", len(reranked_evidence))

            except Exception as e:
                span.record_exception(e)
                # Fallback: keep top evidence by original score
                if state.fusion_evidence:
                    sorted_evidence = sorted(state.fusion_evidence, key=lambda x: x.get("score", 0), reverse=True)
                    state.reranked_evidence = sorted_evidence[:self.config.rerank_k]
                else:
                    state.reranked_evidence = []

        return state

    def draft_answer_node(self, state: AppState) -> AppState:
        """Generate answer with inline citations using reranked evidence"""
        with self.tracer.start_as_current_span("answer.draft") as span:
            span.set_attribute("draft.evidence_count", len(state.reranked_evidence or []))

            try:
                from openai import OpenAI
                client = OpenAI()
                model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

                if not state.reranked_evidence:
                    state.draft_answer = "I don't have enough information to answer this question."
                    return state

                # Prepare evidence context
                evidence_context = ""
                for i, evidence in enumerate(state.reranked_evidence):
                    source = evidence.get("metadata", {}).get("source", "unknown")
                    evidence_context += f"[{i+1}] {evidence['text']}\nSource: {source}\n\n"

                system_prompt = """You are an expert assistant that provides accurate, well-cited answers. Use the provided evidence to answer the user's question.

REQUIREMENTS:
- Base your answer ONLY on the provided evidence
- Include inline citations like [1], [2] for specific claims
- If evidence is insufficient, clearly state what's missing
- Be comprehensive but concise
- Structure your answer clearly with headings if helpful

CITATION FORMAT:
- Use [1], [2], etc. to reference evidence items
- Cite specific claims, not general statements
- Multiple sources can support one claim: [1,2]"""

                user_prompt = f"""Question: {state.user_query}

Evidence:
{evidence_context}

Provide a comprehensive answer with proper citations:"""

                with self.tracer.start_as_current_span("gen_ai.draft_answer") as gen_span:
                    gen_span.set_attribute("gen_ai.system", "openai")
                    gen_span.set_attribute("gen_ai.request.model", model)

                    response = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )

                    try:
                        answer = response.output[0].content[0].text
                    except:
                        answer = str(response)

                    state.draft_answer = answer
                    span.set_attribute("draft.answer_length", len(answer))

            except Exception as e:
                span.record_exception(e)
                state.draft_answer = f"Error generating answer: {e}"

        return state

    def verify_answer_node(self, state: AppState) -> AppState:
        """Verify answer quality and confidence using LLM judge"""
        with self.tracer.start_as_current_span("answer.verify") as span:
            span.set_attribute("verify.answer_length", len(state.draft_answer or ""))

            try:
                from openai import OpenAI
                client = OpenAI()
                model = os.getenv("OPENAI_MODEL", "gpt-5-nano")

                system_prompt = """You are an answer quality evaluator. Assess the answer's quality across multiple dimensions.

Return JSON with this structure:
{
    "coverage": 0.85,  // How well the answer covers the question (0-1)
    "faithfulness": 0.90,  // How faithful the answer is to evidence (0-1)
    "confidence": 0.80,  // Overall confidence in answer quality (0-1)
    "missing_aspects": ["aspect1", "aspect2"],  // What key aspects are missing
    "potential_issues": ["issue1", "issue2"],  // Potential problems or uncertainties
    "reasoning": "Detailed explanation of scores and issues"
}

SCORING GUIDELINES:
- Coverage: Does the answer address all parts of the question?
- Faithfulness: Is the answer supported by the evidence? No hallucination?
- Confidence: Overall reliability for publication (combines coverage + faithfulness + completeness)
- Be strict but fair in scoring"""

                user_prompt = f"""Question: {state.user_query}

Answer to evaluate:
{state.draft_answer}

Evidence that was available:
{chr(10).join([f"- {e['text'][:200]}..." for e in (state.reranked_evidence or [])])}

Evaluate this answer:"""

                with self.tracer.start_as_current_span("gen_ai.verify") as gen_span:
                    gen_span.set_attribute("gen_ai.system", "openai")
                    gen_span.set_attribute("gen_ai.request.model", model)

                    response = client.responses.create(
                        model=model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )

                    try:
                        response_text = response.output[0].content[0].text
                    except:
                        response_text = str(response)

                    # Parse verification result
                    try:
                        verification_data = json.loads(response_text.strip())
                        verification = VerificationResult(**verification_data)
                    except:
                        # Fallback verification
                        verification = VerificationResult(
                            coverage=0.5,
                            faithfulness=0.7,
                            confidence=0.6,
                            missing_aspects=["Unable to assess"],
                            potential_issues=["Verification parsing failed"],
                            reasoning="Automatic verification failed, using conservative scores"
                        )

                    state.verification_result = asdict(verification)

                    span.set_attribute("verify.coverage", verification.coverage)
                    span.set_attribute("verify.faithfulness", verification.faithfulness)
                    span.set_attribute("verify.confidence", verification.confidence)
                    span.set_attribute("verify.missing_count", len(verification.missing_aspects))

            except Exception as e:
                span.record_exception(e)
                # Fallback verification
                state.verification_result = asdict(VerificationResult(
                    coverage=0.3,
                    faithfulness=0.5,
                    confidence=0.4,
                    missing_aspects=["Verification failed"],
                    potential_issues=[f"Error during verification: {e}"],
                    reasoning="Verification process encountered an error"
                ))

        return state

    def threshold_or_escalate_node(self, state: AppState) -> AppState:
        """Check if confidence meets threshold, otherwise escalate with EXA search"""
        with self.tracer.start_as_current_span("gate.threshold") as span:
            verification = state.verification_result or {}
            confidence = verification.get("confidence", 0.0)
            loops_done = getattr(state, "accuracy_loops", 0)

            span.set_attribute("threshold.confidence", confidence)
            span.set_attribute("threshold.target", self.config.accuracy_target)
            span.set_attribute("threshold.loops_done", loops_done)

            if confidence >= self.config.accuracy_target:
                state.threshold_decision = "publish"
                span.set_attribute("threshold.decision", "publish")
            elif loops_done >= self.config.max_loops:
                state.threshold_decision = "hitl_budget_exceeded"
                span.set_attribute("threshold.decision", "hitl_budget_exceeded")
            else:
                # Escalate: try EXA search for fresh sources
                state.threshold_decision = "escalate_exa"
                span.set_attribute("threshold.decision", "escalate_exa")

                try:
                    # Perform EXA search
                    exa_results = exa_search_summary(
                        query=state.user_query,
                        num_results=self.config.exa_num_results
                    )

                    if exa_results["results"]:
                        # Convert EXA results to evidence format
                        exa_evidence = []
                        for result in exa_results["results"]:
                            evidence = EvidenceItem(
                                text=result.text,
                                source=result.url,
                                score=result.score or 0.8,  # Default score if not provided
                                metadata={
                                    "title": result.title,
                                    "domain": result.domain,
                                    "published_date": result.published_date,
                                    "source_type": "exa_web"
                                }
                            )
                            exa_evidence.append(evidence)

                        # Add EXA evidence to fusion evidence for next iteration
                        existing_evidence = [EvidenceItem(**e) for e in (state.fusion_evidence or [])]
                        combined_evidence = existing_evidence + exa_evidence
                        deduplicated = self._mmr_deduplicate(combined_evidence, self.config.fusion_k + self.config.exa_num_results)

                        state.fusion_evidence = [asdict(e) for e in deduplicated]
                        state.exa_results = exa_results
                        state.accuracy_loops = loops_done + 1

                        span.set_attribute("exa.results_count", len(exa_results["results"]))
                        span.set_attribute("exa.domains_count", len(exa_results["domains"]))

                except Exception as e:
                    span.record_exception(e)
                    state.threshold_decision = "hitl_exa_failed"
                    state.exa_error = str(e)

        return state

    def compliance_gate_node(self, state: AppState) -> AppState:
        """Check for compliance issues in the answer"""
        with self.tracer.start_as_current_span("gate.compliance") as span:
            span.set_attribute("compliance.answer_length", len(state.draft_answer or ""))

            compliance_issues = []

            # Check for potential PII patterns
            import re
            pii_patterns = [
                (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN pattern'),
                (r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b', 'Credit card pattern'),
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email address'),
                (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 'Phone number'),
            ]

            answer_text = state.draft_answer or ""
            for pattern, description in pii_patterns:
                if re.search(pattern, answer_text):
                    compliance_issues.append(f"Potential PII detected: {description}")

            # Check for code execution risks
            code_patterns = [
                (r'import\s+os', 'OS module import'),
                (r'subprocess', 'Subprocess usage'),
                (r'eval\s*\(', 'Eval function'),
                (r'exec\s*\(', 'Exec function'),
                (r'__import__', 'Dynamic import'),
            ]

            for pattern, description in code_patterns:
                if re.search(pattern, answer_text, re.IGNORECASE):
                    compliance_issues.append(f"Code execution risk: {description}")

            # Check answer length (simple budget check)
            if len(answer_text) > 10000:
                compliance_issues.append("Answer length exceeds safe limits")

            # Check for certain forbidden domains in sources
            forbidden_domains = ["malware.", "phishing.", "spam."]
            for evidence in (state.reranked_evidence or []):
                source = evidence.get("source", "")
                for domain in forbidden_domains:
                    if domain in source.lower():
                        compliance_issues.append(f"Potentially unsafe source domain: {domain}")

            state.compliance_issues = compliance_issues
            state.compliance_passed = len(compliance_issues) == 0

            span.set_attribute("compliance.issues_count", len(compliance_issues))
            span.set_attribute("compliance.passed", state.compliance_passed)

        return state

    def budget_gate_node(self, state: AppState) -> AppState:
        """Track and enforce budget limits"""
        with self.tracer.start_as_current_span("gate.budget") as span:
            # Initialize budget tracking if not present or None
            if not getattr(state, "budget_used", None):
                state.budget_used = {
                    "tokens": 0,
                    "tool_calls": 0,
                    "seconds": 0,
                    "start_time": time.time()
                }

            # Update budget usage (simplified tracking)
            budget = state.budget_used
            budget["tool_calls"] += 1
            budget["seconds"] = time.time() - budget["start_time"]

            # Estimate token usage (rough)
            total_text = ""
            total_text += state.user_query or ""
            total_text += state.draft_answer or ""
            for evidence in (state.reranked_evidence or []):
                total_text += evidence.get("text", "")

            estimated_tokens = len(total_text.split()) * 1.3  # Rough token estimate
            budget["tokens"] = int(estimated_tokens)

            # Check budget limits
            budget_issues = []
            if budget["tokens"] > self.config.max_tokens:
                budget_issues.append(f"Token limit exceeded: {budget['tokens']}/{self.config.max_tokens}")

            if budget["tool_calls"] > self.config.max_tool_calls:
                budget_issues.append(f"Tool call limit exceeded: {budget['tool_calls']}/{self.config.max_tool_calls}")

            if budget["seconds"] > self.config.max_seconds:
                budget_issues.append(f"Time limit exceeded: {budget['seconds']:.1f}/{self.config.max_seconds}")

            state.budget_issues = budget_issues
            state.budget_exceeded = len(budget_issues) > 0

            span.set_attribute("budget.tokens", budget["tokens"])
            span.set_attribute("budget.tool_calls", budget["tool_calls"])
            span.set_attribute("budget.seconds", budget["seconds"])
            span.set_attribute("budget.exceeded", state.budget_exceeded)

        return state

    def _mmr_deduplicate(self, evidence_list: List[EvidenceItem], target_count: int) -> List[EvidenceItem]:
        """Simple MMR-style deduplication based on text similarity"""
        if len(evidence_list) <= target_count:
            return evidence_list

        # Simple deduplication: remove items with high text overlap
        deduplicated = []
        for evidence in sorted(evidence_list, key=lambda x: x.score, reverse=True):
            is_duplicate = False
            for existing in deduplicated:
                # Simple similarity check
                overlap = len(set(evidence.text.lower().split()) & set(existing.text.lower().split()))
                max_words = max(len(evidence.text.split()), len(existing.text.split()))
                similarity = overlap / max_words if max_words > 0 else 0

                if similarity > 0.7:  # Threshold for considering duplicates
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(evidence)

            if len(deduplicated) >= target_count:
                break

        return deduplicated

# Routing functions for LangGraph
def route_after_threshold_check(state: AppState) -> str:
    """Route based on threshold decision"""
    decision = getattr(state, "threshold_decision", "publish")

    if decision == "publish":
        return "compliance_gate"
    elif decision == "escalate_exa":
        return "rerank_llm"  # Go back to reranking with new evidence
    else:  # HITL scenarios
        return "hitl_accuracy_review"

def route_after_compliance(state: AppState) -> str:
    """Route based on compliance check"""
    if getattr(state, "compliance_passed", True):
        return "budget_gate"
    else:
        return "hitl_compliance_review"

def route_after_budget(state: AppState) -> str:
    """Route based on budget check"""
    if getattr(state, "budget_exceeded", False):
        return "hitl_budget_review"
    else:
        return "finalize_answer"