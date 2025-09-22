"""Self-contained HITL workflow with session management."""

import uuid
from typing import Dict, Optional
from .hitl_workflow import create_hitl_graph
from ..models.hitl_models import HITLState, HITLStage, HITLDecision


class SelfContainedHITLWorkflow:
    """Self-contained workflow manager with session management."""

    def __init__(self):
        self.graph = create_hitl_graph()
        self.sessions: Dict[str, HITLState] = {}

    def start_session(self, user_query: str) -> str:
        """Start a new HITL session and return session ID."""
        session_id = str(uuid.uuid4())

        # Create initial state
        initial_state = HITLState(
            user_query=user_query,
            session_id=session_id,
            stage=HITLStage.GENERATE_CODE
        )

        # Store session
        self.sessions[session_id] = initial_state

        return session_id

    def get_session(self, session_id: str) -> Optional[HITLState]:
        """Get session state by ID."""
        return self.sessions.get(session_id)

    def run_step(self, session_id: str) -> Optional[HITLState]:
        """Run the next step in the workflow for the given session."""
        session = self.get_session(session_id)
        if not session:
            return None

        # Run workflow step
        config = {"configurable": {"thread_id": session_id}}
        updated_state = self.graph.invoke(session, config=config)

        # Update stored session
        self.sessions[session_id] = updated_state

        return updated_state

    def submit_decision(self, session_id: str, decision: HITLDecision, feedback: str = "") -> bool:
        """Submit user decision for a session."""
        session = self.get_session(session_id)
        if not session or not session.approval_needed:
            return False

        # Update session with user decision
        updated_session = session.model_copy(update={
            "user_decision": decision,
            "approval_needed": False
        })

        self.sessions[session_id] = updated_session

        return True