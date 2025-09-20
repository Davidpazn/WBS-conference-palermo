# AI Agents Demo

## A walkthrough on (LangGraph + OpenAI + Pydantic v2 + Letta + Qdrant + OTEL + LangSmith + E2B + MCP)

**What's here:** ship 3 demo notebooks + a tiny full-stack app (FastAPI backend, Next.js frontend) that showcases core agentic patterns, tracing, memory, and evals — Following what's been shown throughout the presentation.

---

## Versions (pinned)

Python: **3.12.11**  
Node (UI): **22.x (LTS-compatible)**  
Key libs (see `environment.yml`):

- langgraph **0.6.7**
- openai **1.108.1**
- pydantic **2.11.9**
- langsmith **0.4.29**
- qdrant-client **1.15.1**
- opentelemetry-\* **1.37.0** (instrumentation **0.58b0**)
- fastapi **0.115.5**, uvicorn **0.36.0**
- letta **0.11.7**, letta-client **0.1.321**
- e2b-code-interpreter **2.0.0**
- jupyterlab **4.4.7**

Frontend (separate `package.json`):

- next **15.5.3**, react **19.0.0**, react-dom **19.0.0**

---

## Repo structure

```
/agents-app
/backend # FastAPI + LangGraph + Letta + Qdrant client
/frontend # Next.js 15 + Prisma (SQLite)
/notebooks
01_one_shot_agent.ipynb
02_rebalancing_agent.ipynb
03_graph_memory_compliance.ipynb
/infra # OTEL config, docker helpers, .env.example
.claude/agents.yaml # Claude Code sub-agents & permissions
environment.yml
Makefile
README.md
```

---

## Quickstart

1. **Create env & kernel**

```bash
make env && make kernel
Frontend deps & DB

bash
Copy code
make frontend-setup
make migrate
Run the stack (dev)

bash
Copy code
make dev          # API (bg) + UI (fg)
# or separately:
make run-api
make dev-ui
Open notebooks

bash
Copy code
make nb1   # one-shot coding agent (E2B)
make nb2   # rebalancing agent + tools + evals
make nb3   # graphs + memory (Letta) + compliance gate
Make targets
text
Copy code
env            Create Conda env from environment.yml
env-update     Update env (prunes extras)
kernel         Register Jupyter kernel
run-api        Start FastAPI (Uvicorn) --reload
frontend-setup Install UI deps (pnpm)
migrate        Prisma migrate dev (SQLite)
dev-ui         Start Next.js dev server
dev            API (bg) + UI (fg)
test           Run pytest -q
notebooks|nb1|nb2|nb3  Open JupyterLab / specific notebook
qdrant-up      Start Qdrant (Docker) on :6333
qdrant-down    Stop & remove Qdrant
clean          Remove __pycache__/pyc
Environment variables
Create ./infra/.env.example and copy to .env files (backend, frontend as needed):

ini
Copy code
# Model / API
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# Tracing (LangSmith)
LANGSMITH_API_KEY=ls-...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=agents-demo

# OpenTelemetry (optional OTLP endpoint)
OTEL_SERVICE_NAME=agents-backend
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc

# Qdrant (local Docker or Cloud)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Letta
LETTA_BASE_URL=http://localhost:8283
LETTA_API_KEY=

# E2B
E2B_API_KEY=

# EXA
EXA_API_KEY=

If using local Qdrant: make qdrant-up (persists to named volume).
```

Claude Code setup

We use .claude/agents.yaml with four focused agents:

Architect (design & schemas), Coder (impl + tests + tracing),

Tester (pytest + LangSmith evals), Reviewer (correctness & spans).

Permissions notes

Use WebFetch(domain:hostname) format (e.g., domain:opentelemetry.io).

Keep Write, Bash gated to avoid accidental destructive ops.

Prefer uv/pip in the Conda env; pnpm via Corepack in frontend.

Backend

Framework: FastAPI with Uvicorn.

Graphs: LangGraph (typed state via Pydantic v2).

Memory: Letta (client + optional local server).

Vector: Qdrant (ingest + query paths).

Tracing: OpenTelemetry spans; LangSmith runs/evals.

Dev endpoints (suggested):

POST /invoke — run graph with streaming

POST /memory/\* — Letta proxy (blocks/agents)

GET /healthz — health check

Frontend

Next.js 15 (App Router) + Prisma (SQLite for speed).

Pages: / (run panel), /agents (create/run), /notebooks (links).

Minimal Run model: id, graphName, status, cost, createdAt.

Notebooks (scope & checkpoints)

01_one_shot_agent.ipynb

ReAct one-shot coding agent → execute in E2B sandbox → auto-tests.

LangSmith trace; verify OTEL spans exist.

02_rebalancing_agent.ipynb

Inputs → plan → trades (typed Pydantic models).

Qdrant rules/doc RAG; LangSmith evals/datasets.

03_graph_memory_compliance.ipynb

Supervisor/subgraphs with Letta memory.

Human-in-the-loop checkpoint; compliance gate; tracing/export.

Today’s checklist

make env && make kernel

.env files set (OpenAI, LangSmith, etc.)

make qdrant-up (if local)

make frontend-setup && make migrate

make dev (API+UI)

Run NB1 → NB2 → NB3 end-to-end

Sanity: traces in LangSmith, OTEL spans visible, tests pass

Troubleshooting

Claude Code WebFetch error → ensure domain: prefix in .claude/agents.yaml.

Prisma migrate fails → delete frontend/prisma/dev.db and re-run make migrate.

No traces → confirm LANGSMITH_TRACING=true, OTLP endpoint reachable.

Qdrant auth → leave QDRANT_API_KEY empty for local Docker; set for Cloud.

License

TBD (Apache-2.0 recommended for demos).
