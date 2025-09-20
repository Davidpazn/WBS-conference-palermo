Perfect—updated to center the **showcase** goal (not just Claude Code productivity) and with a **clear documentation index** Claude can WebFetch anytime.

```markdown
# CLAUDE.md — Agent Flows Showcase (LangGraph • OpenAI • Letta • Qdrant • OTEL • LangSmith • E2B • MCP • Next.js)

> **Purpose:** This repo demonstrates end-to-end **AI agent flows** and how **coding agents** are leveraged with **E2B sandboxes**, **Letta memory**, **Qdrant retrieval**, **LangGraph orchestration**, **OpenTelemetry + LangSmith** tracing/evals, and a small **Next.js** UI.  
> Claude Code should use the **docs index below** (via WebFetch) whenever deeper details are needed.

---

## Repo Facts Claude Should Know

- **Python env:** `conda` env `ai-agents-demo` from `environment.yml` (Python 3.12.x).
- **Backend:** FastAPI in `backend/` with LangGraph nodes, OpenAI SDK, Letta client, Qdrant client, OTEL, LangSmith.
- **Frontend:** Next.js 15 (App Router) in `frontend/` with Prisma (SQLite).
- **Notebooks:** `notebooks/01_*`, `02_*`, `03_*` (E2B coding agent; rebalancing+tools; graphs+memory+compliance).
- **Make:** `make env`, `make run-api`, `make dev-ui`, `make nb1/nb2/nb3`, `make qdrant-up`.

---

## Documentation Index (official links)

**LangGraph**

- Overview: https://www.langchain.com/langgraph
- OSS docs: https://langchain-ai.github.io/langgraph/
- Platform reference: https://docs.langchain.com/langgraph-platform/reference-overview

**OpenAI (Python SDK & Responses)**

- API reference: https://platform.openai.com/docs/api-reference/introduction
- Streaming (Responses): https://platform.openai.com/docs/guides/streaming-responses
- Structured outputs (Responses): https://platform.openai.com/docs/guides/structured-outputs

**Pydantic v2**

- Docs (latest): https://docs.pydantic.dev/latest/
- v1→v2 migration: https://docs.pydantic.dev/latest/migration/

**FastAPI**

- Docs home: https://fastapi.tiangolo.com/
- Tutorial: https://fastapi.tiangolo.com/tutorial/
- Reference: https://fastapi.tiangolo.com/reference/

**OpenTelemetry (Python)**

- Language docs: https://opentelemetry.io/docs/languages/python/
- Python API reference: https://opentelemetry-python.readthedocs.io/

**LangSmith**

- Docs: https://docs.langchain.com/langsmith
- Python client reference: https://docs.smith.langchain.com/reference/python/client/langsmith.client.Client

**Qdrant**

- Docs home: https://qdrant.tech/documentation/
- Local quickstart: https://qdrant.tech/documentation/quickstart/
- Python client docs: https://python-client.qdrant.tech/

**Letta (Agentic Memory)**

- Docs home: https://docs.letta.com/
- Overview: https://docs.letta.com/overview
- Memory guide: https://docs.letta.com/guides/agents/memory
- API reference: https://docs.letta.com/api-reference/overview

**E2B (Secure Code Sandboxes)**

- Docs: https://e2b.dev/docs
- Code Interpreter SDK: https://github.com/e2b-dev/code-interpreter

**Model Context Protocol (MCP)**

- Site: https://modelcontextprotocol.io/
- Tools concept: https://modelcontextprotocol.io/docs/concepts/tools
- Spec: https://modelcontextprotocol.io/specification/2025-06-18

**Next.js (UI)**

- Docs home: https://nextjs.org/docs
- App Router getting started: https://nextjs.org/docs/app/getting-started

---

## Agents & Responsibilities (see `.claude/agents.yaml`)

| Agent     | What it does for the **showcase**                                                                                                                   | Tools                             |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| Architect | Picks the pattern (reactive graph, supervisor/subgraphs), defines Pydantic v2 state, plans tracing/evals, and where memory & vector search plug in. | Read, Write, Grep, WebFetch       |
| Coder     | Implements nodes/tools, FastAPI routes, OTEL + LangSmith, Qdrant ingest/query, Letta calls; adds tests & smoke cells.                               | Read, Write, Grep, WebFetch, Bash |
| Tester    | Pytest suites, LangSmith datasets/evals, minimal e2e.                                                                                               | Read, Write, Grep, WebFetch, Bash |
| Reviewer  | Reviews correctness, latency, spans/attributes, schema contracts, version pins.                                                                     | Read, Grep, WebFetch              |

**Permissions note:** Use `WebFetch(domain:hostname)` (already allow-listed for the doc domains above).

---

## Working Style & Output Rules

1. **Plan → Patch → Test → Run instructions.** Keep patches small and explicit.
2. **Typed everything:** Pydantic v2 models for state, tool I/O, API schemas.
3. **Telemetry-first:** Wrap each node/tool with OTEL spans; attach LangSmith run metadata.
4. **Determinism/idempotence:** Nodes must be deterministic given the same state and inputs.
5. **Secrets:** Use `.env` (see `infra/.env.example`); never commit keys.

---

## Prompt Snippets (copy/paste to kick off tasks)

**A) Scaffold LangGraph app + FastAPI route**
```

Create backend/app/ with:

- state.py: Pydantic v2 BaseModel (AppState) fields: user_query\:str, messages\:list, result\:str|None, cost\:float|None
- nodes.py: plan_node(state), act_node(state) calling OpenAI Responses (structured outputs where useful, retries), observe_node(state)
- graph.py: reactive graph plan→act→observe (+ in-memory checkpointer)
- telemetry.py: OTEL tracer provider (OTLP if env set, console fallback)
- main.py: FastAPI app; /healthz; POST /invoke runs graph once; LangSmith run metadata

Add backend/tests/test_graph.py (happy+error). Show tree and how to run: `make run-api`.

```

**B) Wire Qdrant + rules tool**
```

Add vector.py (client factory from QDRANT_URL/API_KEY) and ingest_rules.py (small seed).
Expose tool query_rules(topic\:str)->list\[str] used by act_node.
Include mocked tests. Provide `make qdrant-up` + sample query.

```

**C) Plug Letta memory**
```

Add memory.py with save_memory(user_id, item), recall_memory(user_id, k=5).
Modify /invoke to save {user_query,result} and include recall in context when env is present; graceful no-op otherwise.
Unit tests mock HTTP client.

```

**D) E2B demo (NB1)**
```

Notebook cells: start E2B session → write+run small Python file → agent writes pytest → run tests → record LangSmith run.
Provide exact cells and a final smoke test cell.

```

**E) Streaming to UI**
```

Implement SSE/streamed chunks from /invoke. Provide minimal Next.js fetch/reader example to render partial tokens and final cost.

```

---

## Review Checklists

**Graph & Tools**
- [ ] Pydantic v2 state & tool I/O; no hidden globals.
- [ ] Retries/timeouts/budgets on OpenAI calls; structured outputs where helpful.
- [ ] Qdrant collection exists; ingest script works locally (Docker) and mock-tested.
- [ ] Letta calls tolerate missing env (skip with log).

**Telemetry & Evals**
- [ ] Spans on each node/tool with attributes: `node`, `model`, `latency_ms`, `tokens_in/out`, `cost`.
- [ ] LangSmith run created with `project`, `graph_name`, `run_id`; at least 2 example evals (NB2).

**API & UI**
- [ ] `/healthz` OK; `/invoke` streams; CORS as needed.
- [ ] UI renders stream; Prisma `Run` persisted.

**Notebooks**
- [ ] “Run All” clean on fresh env; no secrets; final PASS/FAIL cell.

---

## Troubleshooting

- **WebFetch blocked** → add `WebFetch(domain:hostname)` to `.claude/agents.yaml`.
- **No spans** → check `OTEL_*` env; console exporter should still print.
- **No LangSmith traces** → ensure `LANGSMITH_TRACING=true` and API key set.
- **Qdrant conn refused** → `make qdrant-up` or set Cloud URL.
- **Letta down** → skip memory (graceful), log a warning.

---
```

**Docs links verified from official sources**: LangGraph (overview & docs) ([LangChain][1]) • OpenAI API (reference, streaming, structured outputs) ([OpenAI Platform][2]) • Pydantic v2 (docs & migration) ([Pydantic][3]) • FastAPI (docs/tutorial/ref) ([FastAPI][4]) • OTEL Python ([OpenTelemetry][5]) • LangSmith (docs & client ref) ([LangChain Docs][6]) • Qdrant (docs/quickstart/client) ([Qdrant][7]) • Letta (docs/overview/memory/API) ([Letta][8]) • E2B (docs & SDK) ([E2B][9]) • MCP (site/tools/spec) ([Model Context Protocol][10])

If you want this saved to the repo, say the word and I’ll add it as `CLAUDE.md`.

[1]: https://www.langchain.com/langgraph?utm_source=chatgpt.com "LangGraph"
[2]: https://platform.openai.com/docs/api-reference/introduction?utm_source=chatgpt.com "OpenAI API Reference"
[3]: https://docs.pydantic.dev/latest/?utm_source=chatgpt.com "Welcome to Pydantic - Pydantic"
[4]: https://fastapi.tiangolo.com/?utm_source=chatgpt.com "FastAPI"
[5]: https://opentelemetry.io/docs/languages/python/?utm_source=chatgpt.com "Python"
[6]: https://docs.langchain.com/langsmith?utm_source=chatgpt.com "Get started with LangSmith - Docs by LangChain"
[7]: https://qdrant.tech/documentation/?utm_source=chatgpt.com "Qdrant Documentation"
[8]: https://docs.letta.com/?utm_source=chatgpt.com "Letta: Home"
[9]: https://e2b.dev/docs?utm_source=chatgpt.com "E2B - Code Interpreting for AI apps"
[10]: https://modelcontextprotocol.io/?utm_source=chatgpt.com "Model Context Protocol"
