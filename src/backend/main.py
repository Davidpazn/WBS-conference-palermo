# backend/main.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Generator
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from langgraph.types import Command

# Load .env early (no-op if missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from src.backend.app.state import AppState
from src.backend.app.graph import build_graph
from src.infra.telemetry import setup_tracer
from src.backend.app import memory as letta  # for /debug/letta

# -----------------------------------------------------------------------------
# App & middleware
# -----------------------------------------------------------------------------

app = FastAPI(title="Agents Demo API", version=os.getenv("BACKEND_VERSION", "0.1.0"))

# CORS: allow localhost UI by default; override with CORS_ORIGINS env (comma-separated)
origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Telemetry + Graph
# -----------------------------------------------------------------------------

TRACER = setup_tracer(service_name=os.getenv("OTEL_SERVICE_NAME", "agents-backend"))
GRAPH = build_graph()

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class InvokeBody(BaseModel):
    user_id: str
    user_query: str
    meta: Dict[str, Any] = {}
    # For UI convenience; graph controls memory via feature flags/config
    use_memory: Optional[bool] = True
    stream: Optional[bool] = False  # if true, return SSE
    thread_id: Optional[str] = None  # for HITL session persistence

class ResumeBody(BaseModel):
    thread_id: str
    decision: Dict[str, Any]  # Human decision to resume interrupted workflow

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/debug/env")
def debug_env():
    # Never returns secrets; only presence flags
    return {
        "openai_key": "present" if bool(os.getenv("OPENAI_API_KEY")) else "missing",
        "letta_key": "present" if bool(os.getenv("LETTA_API_KEY")) else "missing",
        "use_letta": os.getenv("USE_LETTA", "false"),
        "letta_base": os.getenv("LETTA_BASE_URL", "http://localhost:8283"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "cors_origins": allow_origins,
    }

@app.get("/debug/letta")
def debug_letta():
    # Lightweight connectivity check to Letta (does not expose secrets)
    try:
        return letta.health()  # returns dict with ok/use_letta/base_url
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/invoke")
def invoke(body: InvokeBody):
    """
    Run the LangGraph with HITL support.
    - Returns JSON with final state or interrupt payload for human review.
    - If body.stream=true, returns Server-Sent Events (SSE).
    - Uses thread_id for persistent HITL sessions.
    """
    stream_flag = bool(body.stream)
    thread_id = body.thread_id or f"{body.user_id}_{hash(body.user_query) % 10000}"

    # Build initial state
    state = AppState(
        user_id=body.user_id,
        user_query=body.user_query,
        meta=body.meta or {},
    )

    # Configuration for LangGraph execution
    config = {
        "configurable": {"thread_id": thread_id}
    }

    with TRACER.start_as_current_span("invoke") as span:
        span.set_attribute("user.id", body.user_id)
        span.set_attribute("input.query_len", len(body.user_query))
        span.set_attribute("stream.requested", stream_flag)
        span.set_attribute("thread.id", thread_id)

        try:
            # Run graph with checkpointer for HITL support
            result = GRAPH.invoke(state, config=config)

            # Check if we hit an interrupt (HITL pause)
            if "__interrupt__" in result:
                span.set_attribute("hitl.interrupted", True)
                interrupt_info = result["__interrupt__"]

                payload = {
                    "status": "interrupted",
                    "thread_id": thread_id,
                    "interrupt": interrupt_info,
                    "approval_payload": result.get("approval_payload"),
                    "stage": result.get("hitl_stage", "unknown")
                }

                if stream_flag:
                    def sse() -> Generator[bytes, None, None]:
                        yield b"event: interrupt\n"
                        data = json.dumps(payload, ensure_ascii=False)
                        yield f"data: {data}\n\n".encode("utf-8")
                    return StreamingResponse(sse(), media_type="text/event-stream")

                return JSONResponse(payload, status_code=202)  # Accepted, waiting for input

            # Normal completion
            out_state: AppState = result
            span.set_attribute("hitl.completed", True)

        except Exception as e:
            span.record_exception(e)
            raise HTTPException(status_code=500, detail=f"invoke_failed: {e}") from e

    payload = out_state.model_dump()
    payload["thread_id"] = thread_id
    payload["status"] = "completed"

    if stream_flag:
        def sse() -> Generator[bytes, None, None]:
            yield b"event: result\n"
            data = json.dumps(payload, ensure_ascii=False)
            yield f"data: {data}\n\n".encode("utf-8")
        return StreamingResponse(sse(), media_type="text/event-stream")

    return JSONResponse(payload)


@app.post("/resume")
def resume_hitl(body: ResumeBody):
    """
    Resume an interrupted HITL workflow with human decision.
    """
    config = {
        "configurable": {"thread_id": body.thread_id}
    }

    with TRACER.start_as_current_span("resume_hitl") as span:
        span.set_attribute("thread.id", body.thread_id)
        span.set_attribute("decision.type", body.decision.get("decision", "unknown"))

        try:
            # Resume with human decision
            result = GRAPH.invoke(Command(resume=body.decision), config=config)

            # Check for another interrupt or completion
            if "__interrupt__" in result:
                span.set_attribute("hitl.re_interrupted", True)
                interrupt_info = result["__interrupt__"]

                return JSONResponse({
                    "status": "interrupted",
                    "thread_id": body.thread_id,
                    "interrupt": interrupt_info,
                    "approval_payload": result.get("approval_payload"),
                    "stage": result.get("hitl_stage", "unknown")
                }, status_code=202)

            # Final completion
            out_state: AppState = result
            span.set_attribute("hitl.final_completed", True)

            payload = out_state.model_dump()
            payload["thread_id"] = body.thread_id
            payload["status"] = "completed"

            return JSONResponse(payload)

        except Exception as e:
            span.record_exception(e)
            raise HTTPException(status_code=500, detail=f"resume_failed: {e}") from e


@app.get("/threads/{thread_id}/status")
def get_thread_status(thread_id: str):
    """Get the current status of a HITL thread."""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Check if thread exists and get its state
        state = GRAPH.get_state(config)
        if not state:
            raise HTTPException(status_code=404, detail="Thread not found")

        return {
            "thread_id": thread_id,
            "next": state.next,
            "has_interrupt": bool(state.tasks),
            "values": state.values if hasattr(state, 'values') else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"status_check_failed: {e}") from e

# -----------------------------------------------------------------------------
# Optional: local dev entrypoint (uvicorn)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn  # type: ignore
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
