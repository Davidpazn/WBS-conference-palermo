from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum

class WorkflowStage(str, Enum):
    """Enumeration of workflow stages"""
    INIT = "init"
    QUERY_EXPANSION = "query_expansion"
    RAG_RETRIEVAL = "rag_retrieval"
    EXA_SEARCH = "exa_search"
    FUSION = "fusion"
    RERANKING = "reranking"
    VERIFICATION = "verification"
    THRESHOLD_CHECK = "threshold_check"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    EXECUTION = "execution"
    EXECUTION_REVIEW = "execution_review"
    COMPLIANCE = "compliance"
    BUDGET_CHECK = "budget_check"
    COMPLETE = "complete"
    ERROR = "error"

class HitlStage(str, Enum):
    """HITL interaction stages"""
    CODE_REVIEW = "code_review"
    EXECUTION_REVIEW = "execution_review"
    COMPLIANCE_REVIEW = "compliance_review"
    FINAL_APPROVAL = "final_approval"

class ThresholdDecision(str, Enum):
    """Threshold routing decisions"""
    PUBLISH = "publish"
    ESCALATE_EXA = "escalate_exa"
    ESCALATE_HITL = "escalate_hitl"
    ESCALATE_COMPLIANCE = "escalate_compliance"

class AppState(BaseModel):
    """Comprehensive application state for multiagent workflows"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')

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
    hitl_stage: Optional[HitlStage] = None

    # Accuracy-driven workflow fields
    query_variations: List[Dict[str, Any]] = Field(default_factory=list)
    rag_results: List[Dict[str, Any]] = Field(default_factory=list)
    exa_results: Optional[Dict[str, Any]] = None  # Changed to Dict to match EXA search summary structure
    fused_results: List[Dict[str, Any]] = Field(default_factory=list)
    reranked_results: List[Dict[str, Any]] = Field(default_factory=list)
    reranked_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    fusion_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    query_results: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    verification_result: Optional[Dict[str, Any]] = None
    threshold_decision: Optional[str] = None  # Changed from ThresholdDecision enum
    confidence_score: float = 0.0
    accuracy_loops: int = 0

    # Compliance and budget tracking
    compliance_check: Optional[Dict[str, Any]] = None
    compliance_issues: List[str] = Field(default_factory=list)
    compliance_passed: bool = True
    budget_status: Optional[Dict[str, Any]] = None
    budget_used: Optional[Dict[str, Any]] = None
    budget_issues: List[str] = Field(default_factory=list)
    budget_exceeded: bool = False

    # Answer generation
    draft_answer: Optional[str] = None
    exa_error: Optional[str] = None

    # Telemetry and cost tracking
    trace_id: Optional[str] = None
    span_context: Optional[Dict[str, str]] = None
    total_cost: float = 0.0
    token_usage: Dict[str, int] = Field(default_factory=dict)

    # Workflow control fields
    current_stage: WorkflowStage = WorkflowStage.INIT
    workflow_history: List[WorkflowStage] = Field(default_factory=list)
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Diagram generation support
    diagram_spec: Optional[Dict[str, Any]] = None
    diagram_result: Optional[Dict[str, Any]] = None
    proposed_diagram: Optional[Dict[str, Any]] = None
    diagram_proposal_success: bool = False
    diagram_proposal_error: Optional[str] = None
    diagram_success: bool = False
    diagram_error: Optional[str] = None

    # Memory persistence
    memory_saved: bool = False

    # Advanced HITL fields
    human_edited: bool = False
    additional_search_query: Optional[str] = None
    web_search_query: Optional[str] = None
    code_generation_task: Optional[str] = None
    tool_selection: Optional[Dict[str, Any]] = None
    tool_execution_results: List[Dict[str, Any]] = Field(default_factory=list)
    final_decision: Optional[Dict[str, Any]] = None

    # HITL loop tracking fields
    hitl_loop_count: int = 0
    tool_loop_count: int = 0
    final_loop_count: int = 0
    hitl_actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    tool_selections_history: List[Dict[str, Any]] = Field(default_factory=list)
    hitl_stage: Optional[str] = None

    @field_validator('confidence_score')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0.0 and 1.0')
        return v

    @field_validator('code_retries', 'retry_count')
    @classmethod
    def validate_retries(cls, v):
        if v < 0:
            raise ValueError('Retry counts cannot be negative')
        return v

    def add_stage_to_history(self, stage: WorkflowStage):
        """Add stage to workflow history and update current stage"""
        if self.current_stage != stage:
            self.workflow_history.append(self.current_stage)
            self.current_stage = stage

    def increment_retry(self) -> bool:
        """Increment retry count and return True if under limit"""
        self.retry_count += 1
        return self.retry_count <= self.max_retries
