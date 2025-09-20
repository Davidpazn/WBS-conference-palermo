Awesome — here’s a crisp, end-to-end action plan with **stage order**, plus **To-Do lists** for **NB1, NB2, NB3**, **Backend**, **Infra**, and **UI**, and a clear list of **all API keys** you’ll need (with exact env var names).

---

# Master order (dependency-aware)

1. **Infra baseline** → env, `.env` files, Docker bits (Qdrant optional), OTEL/LangSmith wiring.
2. **Backend skeleton** → FastAPI + LangGraph + Telemetry stubs + `/invoke` + streaming.
3. **UI skeleton** → Next.js page to call `/invoke` and render stream.
4. **NB1** (E2B one-shot coding agent) → prove sandbox + tests + traces.
5. **NB2** (Rebalancing + Qdrant + evals) → vector rules tool + LangSmith dataset.
6. **NB3** (Graphs + Letta memory + compliance gate) → supervisor/subgraphs + HITL checkpoint.
7. **Polish** → run full demo path, verify spans/evals, tighten prompts & limits.

---

# To-Do — Infra

- [ ] **Conda env:** `make env && make kernel` (uses `environment.yml` we pinned).
- [ ] **Env files:** create `infra/.env.example` → copy to:

  - [ ] `backend/.env` (server)
  - [ ] `frontend/.env.local` (UI)

- [ ] **Populate keys** (see “API Keys” below).
- [ ] **Qdrant (optional local):** `make qdrant-up` (Docker).
- [ ] **OTEL exporter (optional):**

  - [ ] If you have an OTLP endpoint: set `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`.
  - [ ] Else console exporter will print spans to stdout.

- [ ] **LangSmith tracing:** set `LANGSMITH_TRACING=true` and `LANGSMITH_PROJECT=agents-demo`.
- [ ] **Sanity:** `echo $OPENAI_API_KEY` etc. from within the env (`conda run -n ai-agents-demo env | grep -E 'OPENAI|LANGSMITH|QDRANT|LETTA|E2B|OTEL'`).

**Definition of done (Infra):** `.env` present with all required keys; `qdrant` container reachable on `:6333` (if using local); `make run-api` prints OTEL console spans.

---

# To-Do — Backend (FastAPI + LangGraph + Telemetry)

**Files/dirs**

- [ ] `backend/app/state.py`

  - Pydantic v2 `AppState(BaseModel)`: `user_query:str`, `messages:list[dict] = []`, `result:str|None = None`, `cost:float|None = None`, `meta:dict = {}`

- [ ] `backend/app/nodes.py`

  - `plan_node(state)`: adds brief plan → `messages`.
  - `act_node(state)`: OpenAI call (Responses API), optional structured output; retries/backoff; budget/tokens in `meta`.
  - `observe_node(state)`: summarize/commit result; update `cost`.

- [ ] `backend/app/graph.py`

  - Build reactive graph `plan → act → observe`; in-memory checkpointer stub.

- [ ] `backend/app/telemetry.py`

  - OTEL tracer/init; OTLP exporter if env set, else console; common attributes.

- [ ] `backend/app/vector.py`

  - Qdrant client factory; `ensure_rules_collection()`, `query_rules(topic:str)->list[str]`.

- [ ] `backend/app/ingest_rules.py`

  - CLI to upsert a few finance/compliance rules (ids, text, tags).

- [ ] `backend/app/memory.py`

  - Letta client helpers: `save_memory(user_id,item)`, `recall_memory(user_id,k=5)`; graceful no-op if no LETTA envs.

- [ ] `backend/main.py`

  - FastAPI app with:

    - `GET /healthz` → `{status:"ok"}`
    - `POST /invoke` → run the graph **once**, **stream** tokens via SSE or chunked responses; attach LangSmith run metadata; optionally call memory helpers.
    - (later) `POST /memory/*` if you expose direct Letta ops for NB3.

- [ ] `backend/tests/test_graph.py`

  - Happy path (returns result + spans recorded).
  - Error path (model failure → retries → 4xx/5xx handled).
  - Qdrant tool mock test.
  - Letta client mock test.

**Commands**

- [ ] `make run-api` → verify `/healthz` and streaming on `/invoke`.
- [ ] `python backend/app/ingest_rules.py` (if using Qdrant).
- [ ] `make test`.

**Definition of done (Backend):** `/invoke` streams; nodes instrumented with spans; optional Qdrant tool works; memory helpers no-op cleanly when unset; tests pass.

---

# To-Do — UI (Next.js 15 + Prisma SQLite)

**Setup**

- [ ] `make frontend-setup` (pnpm deps).
- [ ] `make migrate` (creates initial SQLite schema).

**Files**

- [ ] `frontend/prisma/schema.prisma`

  - Model `Run { id String @id @default(cuid())  graphName String  status String  cost Float?  createdAt DateTime @default(now()) }`

- [ ] `frontend/app/page.tsx` (Home / Run panel)

  - Textarea for prompt; button to call backend `/invoke`; render streamed tokens; final cost/status.

- [ ] `frontend/app/agents/page.tsx` (optional)

  - Simple form to simulate user_id and memory save toggle.

- [ ] `frontend/app/api/run/route.ts` (optional server action wrapper)

  - Proxy to `BACKEND_URL` env for local/prod flexibility.

- [ ] `frontend/lib/stream.ts`

  - Helper to consume SSE/stream.

- [ ] `frontend/.env.local`

  - `NEXT_PUBLIC_BACKEND_URL=http://localhost:8000`

**Commands**

- [ ] `make dev-ui` → open `http://localhost:3000/` and run a prompt.
- [ ] Confirm a `Run` row is saved post-completion.

**Definition of done (UI):** Streaming renders without console errors; final status saved to Prisma; “Run again” works.

---

# To-Do — NB1 (One-shot Coding Agent in **E2B**)

**Goal:** Show a coding agent that **writes code**, **executes it in a secure sandbox**, **writes tests**, and **runs them**; all traced.

**Cells (exact flow)**

1. [ ] **Init**: import; read `E2B_API_KEY` from env; assert present.
2. [ ] **Start session**: create Code Interpreter session; print session id.
3. [ ] **Agent propose code**: small, deterministic function spec (e.g., `moving_average(prices, window)`); save file in the sandbox.
4. [ ] **Run code**: execute; capture stdout/stderr; assert no error.
5. [ ] **Agent propose tests**: generate `test_moving_average.py` (pytest).
6. [ ] **Run tests**: expect all pass; capture output.
7. [ ] **LangSmith trace**: wrap steps in a run; record metadata (tokens, cost if model used).
8. [ ] **Smoke cell**: `assert` on success markers (e.g., `collected 3 items`, `3 passed`).

**Definition of done (NB1):** E2B session opens; code + tests run successfully; trace visible in LangSmith; smoke cell passes.

---

# To-Do — NB2 (Rebalancing Agent + **Qdrant** + Evals)

**Goal:** Typed plan→act→observe agent that outputs **Trades** given a **Portfolio** and **Constraints**, using **Qdrant rules retrieval**; evaluate on 2–3 cases with **LangSmith**.

**Prep**

- [ ] Ingest rules: `python backend/app/ingest_rules.py` (or point to Cloud).
- [ ] Dataset: create small LangSmith dataset programmatically from the notebook.

**Cells**

1. [ ] **Models**: Pydantic v2 `Portfolio`, `Constraint`, `Trade`, `Plan`, `Observation`.
2. [ ] **Agent**: function that calls Qdrant `query_rules(topic)` → include retrieved snippets in prompt → produce `Trades` (structured output).
3. [ ] **Eval set**: 2–3 portfolios with constraints (e.g., target allocations, max single-asset %, do-not-trade list).
4. [ ] **Evaluator**: simple correctness checks (sum to \~100%, no forbidden trades) + textual rationale quality.
5. [ ] **Run eval**: execute all cases; push results to LangSmith; print a summary table.
6. [ ] **Smoke cell**: assert all cases meet constraints and eval returns ≥ threshold.

**Definition of done (NB2):** Rules retrieved from Qdrant; valid `Trades` returned; LangSmith dataset/evals recorded; smoke passes.

---

# To-Do — NB3 (Graphs + **Letta** Memory + Compliance Gate)

**Goal:** Showcase **supervisor + subgraphs** with **agentic memory** (Letta) and a **final compliance gate**; include **human-in-the-loop** (HITL) checkpoint.

**Cells**

1. [ ] **Graph sketch**: supervisor routes between `research_subgraph`, `draft_subgraph`, `finalize_subgraph`.
2. [ ] **Letta memory**: on each iteration, `save_memory(user_id,{query,result})` and `recall_memory(user_id,k=3)` → appended to context; skip gracefully if no LETTA env.
3. [ ] **HITL**: synthetic “approval” step (simulated button/flag in notebook) that can modify state or stop run.
4. [ ] **Compliance gate**: deterministic rule check node; if fail, return actionable error.
5. [ ] **Telemetry**: spans across supervisor and subgraphs; LangSmith run with `graph_name` metadata.
6. [ ] **Smoke cell**: run once with memory **off**, then **on**; assert the “on” run incorporates prior memory.

**Definition of done (NB3):** Memory changes behavior; compliance gate blocks violations; spans show supervisor/subgraphs; smoke passes.

---

# API Keys & Config (exact env vars)

| Service                | Required?                          | Env var(s)                                                                                            | Notes                                                                   |
| ---------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **OpenAI**             | **Yes**                            | `OPENAI_API_KEY` (`OPENAI_BASE_URL` optional)                                                         | Used by backend nodes & notebooks.                                      |
| **LangSmith**          | **Recommended** (for traces/evals) | `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=agents-demo`                        | Enables run tracking & eval datasets.                                   |
| **E2B**                | **Yes** for NB1                    | `E2B_API_KEY`                                                                                         | Needed to open a sandbox session.                                       |
| **Qdrant (Cloud)**     | Optional (Local needs none)        | `QDRANT_URL`, `QDRANT_API_KEY`                                                                        | If running local Docker: `QDRANT_URL=http://localhost:6333` and no key. |
| **Letta**              | Optional (for NB3 memory)          | `LETTA_BASE_URL`, `LETTA_API_KEY`                                                                     | If not set, memory helpers no-op.                                       |
| **OpenTelemetry OTLP** | Optional                           | `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`, `OTEL_SERVICE_NAME=agents-backend` | If unset, console exporter prints spans.                                |
| **UI ↔ Backend URL**   | Yes (UI)                           | `NEXT_PUBLIC_BACKEND_URL`                                                                             | e.g., `http://localhost:8000`.                                          |

**Where to put them**

- **Backend:** `backend/.env` (loaded by `python-dotenv` or your app init).
- **UI:** `frontend/.env.local`.
- **Notebooks:** rely on inherited shell env (start Jupyter from the Conda env).

**Minimal backend `.env` example**

```
OPENAI_API_KEY=sk-***
LANGSMITH_API_KEY=ls-***        # optional but recommended
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=agents-demo

E2B_API_KEY=e2b_***             # needed for NB1 if the backend triggers E2B
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
LETTA_BASE_URL=
LETTA_API_KEY=
OTEL_SERVICE_NAME=agents-backend
# OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
# OTEL_EXPORTER_OTLP_PROTOCOL=grpc
```

---

# Quick smoke tests (per stage)

- **Backend up:** `make run-api` → `GET /healthz` returns ok; POST `/invoke` streams tokens.
- **UI up:** `make dev-ui` → submit a prompt, see streaming output, a `Run` saved in DB.
- **NB1:** open `make nb1` → run all → 100% tests pass in E2B; LangSmith shows one run.
- **NB2:** `make nb2` → eval table shows all pass; Qdrant queries return snippets.
- **NB3:** `make nb3` → second run (with memory) uses recalled context; compliance gate enforces rules.

---
