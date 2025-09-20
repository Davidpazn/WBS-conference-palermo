from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class AppState(BaseModel):
    user_id: str
    user_query: str
    messages: List[Dict] = Field(default_factory=list)
    result: Optional[str] = None
    trades: Optional[list] = None
    meta: Dict = Field(default_factory=dict)

    # Long-term memory integration (Letta)
    recalled: List[Dict] = Field(default_factory=list)
    memory_item: Optional[Dict] = None

    # E2B sandbox integration
    generated_code: Optional[str] = None
    sandbox_execution: Optional[Dict[str, Any]] = None
    code_retries: int = 0

    # Human-in-the-loop (HITL) support
    approval_required: bool = False
    approval_payload: Optional[Dict[str, Any]] = None
    approval_decision: Optional[Dict[str, Any]] = None
    hitl_stage: Optional[str] = None  # "code_review", "execution_approval", etc.

    # Telemetry and cost tracking
    trace_id: Optional[str] = None
    span_context: Optional[Dict[str, str]] = None
    total_cost: float = 0.0
    token_usage: Dict[str, int] = Field(default_factory=dict)
