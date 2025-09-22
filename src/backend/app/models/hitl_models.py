"""HITL workflow models and state definitions."""

import uuid
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class HITLDecision(str, Enum):
    """HITL decision options."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


class HITLStage(str, Enum):
    """HITL workflow stages."""
    GENERATE_CODE = "generate_code"
    REVIEW_CODE = "review_code"
    EXECUTE_CODE = "execute_code"
    SAVE_ARTIFACTS = "save_artifacts"
    COMPLETE = "complete"


class ApprovalPayload(BaseModel):
    """Payload for approval requests."""
    session_id: str = Field(..., min_length=1)
    stage: HITLStage
    content: str
    message: str

    @field_validator('session_id')
    @classmethod
    def session_id_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('session_id cannot be empty')
        return v


class DecisionPayload(BaseModel):
    """Payload for user decisions."""
    session_id: str
    decision: HITLDecision
    feedback: Optional[str] = None


class HITLState(BaseModel):
    """HITL workflow state."""
    user_query: str = Field(..., min_length=1)
    stage: HITLStage = HITLStage.GENERATE_CODE
    session_id: str = Field(..., min_length=1)
    generated_code: Optional[str] = None
    review_comments: List[str] = Field(default_factory=list)
    execution_result: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    approval_needed: bool = False
    user_decision: Optional[HITLDecision] = None

    @field_validator('user_query')
    @classmethod
    def user_query_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('user_query cannot be empty')
        return v

    @field_validator('session_id')
    @classmethod
    def session_id_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('session_id cannot be empty')
        return v


def create_hitl_session(user_query: str) -> HITLState:
    """Create a new HITL session and return initial state."""
    session_id = f"hitl_{uuid.uuid4().hex[:8]}"

    return HITLState(
        user_query=user_query,
        stage=HITLStage.GENERATE_CODE,
        session_id=session_id
    )