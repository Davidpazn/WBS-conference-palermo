# Backend Application Documentation

This document provides comprehensive documentation for the `src/backend/app/` folder structure and all its components. The backend application implements a sophisticated AI agent workflow system with Human-in-the-Loop (HITL) capabilities, E2B sandboxed code execution, Letta memory integration, OpenTelemetry tracing, and LangSmith evaluation.

## Overview

The `src/backend/app` folder contains a comprehensive AI agent framework built around LangGraph orchestration, supporting Human-in-the-Loop (HITL) workflows, E2B sandbox execution, OpenTelemetry tracing, and integrations with OpenAI, Letta (memory), and Qdrant (vector search). This application demonstrates end-to-end AI agent flows for coding tasks, financial document analysis, and compliance checking.

## Architecture Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Graph Nodes   │    │  State Manager   │    │   Integrations  │
│                 │    │                  │    │                 │
│ • plan_node     │◄──►│   AppState       │◄──►│ • OpenAI API    │
│ • act_node      │    │ (Pydantic v2)    │    │ • E2B Sandboxes │
│ • observe_node  │    │                  │    │ • Letta Memory  │
│ • hitl_nodes    │    │ • user_query     │    │ • Qdrant Vector │
│                 │    │ • messages       │    │ • LangSmith     │
└─────────────────┘    │ • generated_code │    └─────────────────┘
                       │ • approval_*     │
┌─────────────────┐    │ • sandbox_*      │    ┌─────────────────┐
│   Telemetry     │    │ • memory_*       │    │   Compliance    │
│                 │◄──►│ • meta/trace     │◄──►│                 │
│ • OpenTelemetry │    └──────────────────┘    │ • Rule Engine   │
│ • LangSmith     │                            │ • Policy Check  │
│ • Spans/Metrics │    ┌──────────────────┐    │ • Violation Log │
│                 │    │  LangGraph Core  │    │                 │
└─────────────────┘    │                  │    └─────────────────┘
                       │ • StateGraph     │
                       │ • Checkpointer   │
                       │ • Conditional    │
                       │   Routing        │
                       └──────────────────┘
```

The backend application is designed as a showcase for end-to-end AI agent flows with the following key features:

- **LangGraph-based workflows** with conditional routing and checkpointing
- **Human-in-the-Loop (HITL)** capabilities for code review and approval
- **E2B sandbox integration** for secure code execution
- **Letta memory system** for long-term memory management
- **OpenTelemetry + LangSmith** for comprehensive tracing and evaluation
- **Qdrant vector database** integration for document retrieval
- **Compliance framework** for enforcing business rules and constraints
- **FastAPI integration** ready (via the main graph execution)

## Folder Structure

```
src/backend/app/
├── __init__.py                 # Package initialization and exports
├── state.py                    # Pydantic state models for LangGraph
├── graph.py                    # Main LangGraph workflow definition
├── nodes.py                    # Core workflow nodes (plan, act, observe)
├── nodes_memory.py             # Memory-specific nodes (recall, save)
├── hitl_nodes.py              # Human-in-the-loop workflow nodes
├── memory.py                   # Letta memory integration
├── utils.py                    # Utility functions for data conversion
├── compliance.py               # Compliance framework implementation
├── agents/                     # Agent implementations
│   ├── __init__.py
│   ├── coding_agent.py        # Basic coding agent
│   ├── hitl_workflow.py       # HITL workflow management
│   └── self_contained_workflow.py  # Standalone workflow manager
├── e2b/                       # E2B sandbox integration
│   ├── __init__.py
│   ├── sandbox.py             # Sandbox lifecycle management
│   ├── execution.py           # Code execution utilities
│   └── artifacts.py           # File and artifact management
├── llm/                       # LLM integration
│   ├── __init__.py
│   ├── openai_client.py       # OpenAI client and code generation
│   └── code_extraction.py     # Code block parsing utilities
├── models/                    # Data models
│   ├── __init__.py
│   ├── hitl_models.py         # HITL workflow state models
│   └── execution_models.py    # Execution-related models
├── telemetry/                 # Observability and tracing
│   ├── __init__.py
│   ├── tracing.py             # OpenTelemetry configuration
│   └── langsmith.py           # LangSmith integration
├── tools/                     # External tools integration
│   ├── __init__.py
│   └── qdrant_admin.py        # Qdrant vector database utilities
└── ingest/                    # Data ingestion utilities
    ├── __init__.py
    ├── config.py              # Ingestion configuration
    ├── edgar_utils.py         # SEC EDGAR data utilities
    ├── company_index.py       # Company data indexing
    ├── chunkers.py            # Text chunking utilities
    ├── linker.py              # Link generation utilities
    └── examples_ingest_5_companies.py  # Example ingestion scripts
```

## Core Components

### 1. State Management (`state.py`)

The `AppState` class is the central Pydantic model that defines the workflow state:

```python
class AppState(BaseModel):
    user_id: str
    user_query: str
    messages: List[Dict] = Field(default_factory=list)
    result: Optional[str] = None

    # Memory integration
    recalled: List[Dict] = Field(default_factory=list)
    memory_item: Optional[Dict] = None

    # E2B sandbox integration
    generated_code: Optional[str] = None
    sandbox_execution: Optional[Dict[str, Any]] = None
    code_retries: int = 0

    # HITL support
    approval_required: bool = False
    approval_payload: Optional[Dict[str, Any]] = None
    approval_decision: Optional[Dict[str, Any]] = None
    hitl_stage: Optional[str] = None

    # Telemetry
    trace_id: Optional[str] = None
    span_context: Optional[Dict[str, str]] = None
    total_cost: float = 0.0
    token_usage: Dict[str, int] = Field(default_factory=dict)
```

**Key Features:**
- Comprehensive state tracking for all workflow components
- Memory integration fields for Letta
- E2B sandbox execution tracking
- HITL workflow state management
- Built-in telemetry and cost tracking

### 2. Workflow Orchestration (`graph.py`)

The main LangGraph workflow supports both HITL coding workflows and legacy plan-act-observe patterns:

```python
def build_graph():
    """Build LangGraph with HITL support for coding agent workflow."""
    g = StateGraph(AppState)

    # Memory integration
    g.add_node("recall", recall_memory_node)

    # HITL Coding Agent Flow
    g.add_node("code_generation", code_generation_node)
    g.add_node("code_review", hitl_code_review_node)
    g.add_node("execute_code", sandbox_execution_node)
    g.add_node("execution_review", hitl_execution_review_node)

    # Memory write-back
    g.add_node("save", save_memory_node)
```

**Key Features:**
- Configurable workflow routing via environment variables
- Built-in checkpointing with MemorySaver
- Conditional routing based on HITL decisions
- Memory recall at start and save at completion

### 3. Human-in-the-Loop Nodes (`hitl_nodes.py`)

Implements sophisticated HITL capabilities with LangGraph's dynamic interrupts:

#### `code_generation_node(state: AppState) -> AppState`
- Uses OpenAI Responses API for structured code generation
- Implements comprehensive tracing with GenAI semantic conventions
- Extracts code from fenced blocks using regex parsing
- Handles retries and error cases gracefully

#### `hitl_code_review_node(state: AppState) -> AppState`
- Creates dynamic interrupts for human review
- Packages code and context for human decision-making
- Supports approve/edit/reject workflows
- Integrates with LangGraph's checkpoint system

#### `sandbox_execution_node(state: AppState) -> AppState`
- Executes code in E2B sandboxes with full isolation
- Implements retry logic for failed executions
- Captures stdout/stderr and execution metadata
- Supports both temporary and persistent sandboxes

**Key Features:**
- Dynamic interrupts with LangGraph's `interrupt()` function
- Comprehensive error handling and recovery
- Full OpenTelemetry tracing integration
- Support for iterative code improvement workflows

### 4. Memory Integration (`memory.py`, `nodes_memory.py`)

Letta integration provides persistent memory capabilities:

#### Memory Client (`memory.py`)
```python
def recall(agent_id: str, query: str, k: int = 3) -> List[Dict]:
    """Recall relevant memory items using Letta client."""

def save(agent_id: str, item: Dict) -> bool:
    """Save memory item to agent's archival memory."""
```

#### Memory Nodes (`nodes_memory.py`)
```python
def recall_memory_node(state: AppState) -> AppState:
    """Long-term memory recall adding top-k items to state."""

def save_memory_node(state: AppState) -> AppState:
    """Long-term memory write-back for conversation persistence."""
```

**Key Features:**
- Graceful degradation when Letta is unavailable
- Automatic agent creation and management
- Version-agnostic SDK support with normalization utilities
- Structured memory items with metadata and tags

### 5. E2B Sandbox Integration (`e2b/`)

#### Sandbox Management (`sandbox.py`)
- Persistent sandbox creation with configurable timeouts
- Sandbox listing and lifecycle management
- Bulk sandbox cleanup utilities

#### Code Execution (`execution.py`)
```python
def run_in_e2b(code: str) -> dict:
    """Create temporary sandbox and execute code."""

def summarize_execution(execution) -> dict:
    """Extract stdout/stderr from execution results."""
```

#### Artifact Management (`artifacts.py`)
- File tree listing and exploration
- TAR archive creation and download
- Base64 encoding for file transfers
- Recursive directory mirroring

**Key Features:**
- Support for both temporary and persistent sandboxes
- Comprehensive file management capabilities
- Error handling and graceful degradation
- SDK version compatibility layer

### 6. LLM Integration (`llm/`)

#### OpenAI Client (`openai_client.py`)
```python
def llm_generate_code(task: str, max_retries: int = 3) -> str:
    """Generate code using OpenAI with full tracing."""
```

**Key Features:**
- Exponential backoff retry logic
- Comprehensive OpenTelemetry tracing
- Token usage and cost tracking
- Configurable model selection

#### Code Extraction (`code_extraction.py`)
- Robust fenced code block parsing
- Multi-language support
- Indentation normalization
- Concatenation of multiple code blocks

### 7. Telemetry and Observability (`telemetry/`)

#### OpenTelemetry Integration (`tracing.py`)
```python
def setup_telemetry() -> Tuple[trace.Tracer, Optional[LangSmithClient]]:
    """Configure OpenTelemetry + LangSmith tracing."""
```

**Features:**
- OTLP exporter configuration
- LangSmith integration for evaluation
- GenAI semantic conventions
- Comprehensive span attribute helpers
- Graceful fallback to console exporter

#### LangSmith Integration (`langsmith.py`)
- Run creation and management
- Evaluation dataset integration
- Error tracking and metrics

### 8. Compliance Framework (`compliance.py`)

Implements a comprehensive compliance gate system:

```python
class ComplianceGate:
    def check_state(self, state: Dict[str, Any]) -> ComplianceResult:
        """Main compliance check for agent state."""
```

**Compliance Categories:**
- **Model Controls (CMP-MOD-*)**: Model allowlists, token limits, cost budgets
- **Tool Safety (CMP-TOOL-*)**: Domain restrictions, shell command filtering
- **Memory Privacy (CMP-MEM-*)**: PII detection, secret leakage prevention
- **Output Quality (CMP-OUT-*)**: Disclaimers, banned phrases, citation requirements
- **Trading Constraints (CMP-TRD-*)**: Asset weight limits, leverage controls

**Key Features:**
- YAML-based configuration
- Action types: BLOCK, WARN, FIX
- Automatic content fixing capabilities
- Comprehensive violation logging

### 9. Vector Database Integration (`tools/qdrant_admin.py`)

Qdrant integration for RAG capabilities:

```python
def ensure_collection_edgar(client: QdrantClient, name: str, vector_size: int):
    """Create EDGAR collection with payload indexes."""

def search_dense_by_text(client: QdrantClient, name: str, query_text: str):
    """Text-based dense vector search with automatic embedding."""
```

**Features:**
- Collection management with automatic indexing
- Metadata filtering (CIK, form types, dates)
- Full-text search capabilities
- Batch ingestion utilities

### 10. Data Ingestion (`ingest/`)

SEC EDGAR data ingestion pipeline:

#### Configuration (`config.py`)
- SEC API endpoints and rate limiting
- Embedding model configuration
- Chunking parameters
- User agent and compliance headers

#### Utilities (`edgar_utils.py`)
- SEC submissions API integration
- HTML content extraction and cleaning
- Document parsing and structuring

**Key Features:**
- Respectful API usage with rate limiting
- Robust HTML parsing and cleaning
- Configurable chunking strategies
- Professional link generation

## Agent Implementations (`agents/`)

### Basic Coding Agent (`coding_agent.py`)
Simple wrapper around the LLM code generation:

```python
def coding_agent(task: str) -> str:
    """Simple coding agent that generates code for a given task."""
    return llm_generate_code(task)
```

### HITL Workflow (`hitl_workflow.py`)
Full-featured HITL workflow with LangGraph integration:

- **Code generation** with OpenAI Responses API
- **Automated code review** with quality checks
- **E2B execution** with error handling
- **Artifact creation** with TAR packaging
- **Session management** with checkpointing

### Self-Contained Workflow (`self_contained_workflow.py`)
Standalone workflow manager for simpler use cases:

```python
class SelfContainedHITLWorkflow:
    def start_session(self, user_query: str) -> str:
    def run_step(self, session_id: str) -> Optional[HITLState]:
    def submit_decision(self, session_id: str, decision: HITLDecision) -> bool:
```

## Data Models (`models/`)

### HITL Models (`hitl_models.py`)
```python
class HITLState(BaseModel):
    user_query: str
    stage: HITLStage
    session_id: str
    generated_code: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
```

**Features:**
- Comprehensive validation with Pydantic v2
- Enum-based stage management
- Session lifecycle tracking

### Execution Models (`execution_models.py`)
```python
class FileEntry(BaseModel):
    name: str
    path: str
    content: Optional[str] = None
    size: Optional[int] = None
```

## Key Design Patterns

### 1. Graceful Degradation
All external services (Letta, E2B, LangSmith) are designed to gracefully degrade:
- Memory operations continue without Letta
- Execution falls back to local environments
- Tracing defaults to console output

### 2. Comprehensive Tracing
Every operation includes OpenTelemetry spans with:
- GenAI semantic conventions for LLM calls
- Custom attributes for HITL and sandbox operations
- Error recording and status tracking
- Cost and token usage metrics

### 3. Type Safety
Extensive use of Pydantic v2 for:
- State validation and serialization
- API contract enforcement
- Configuration management
- Error handling and validation

### 4. Configuration-Driven
Environment variable configuration for:
- Model selection and parameters
- Service endpoints and credentials
- Feature flags and behavior toggles
- Performance tuning parameters

## Integration Points

### FastAPI Integration
The graph can be easily integrated with FastAPI routes:

```python
@app.post("/invoke")
async def invoke_workflow(request: WorkflowRequest):
    graph = build_graph()
    result = graph.invoke(request.to_app_state())
    return result
```

### Streaming Support
The architecture supports streaming via:
- LangGraph's streaming capabilities
- OpenAI's streaming API integration
- Real-time HITL interaction protocols

### External Tool Integration
Easy integration of new tools via:
- Standardized tool interfaces
- Compliance framework integration
- Telemetry and error handling patterns

## Security and Compliance

### Sandbox Isolation
- All code execution in isolated E2B environments
- No access to host filesystem or network
- Configurable timeout and resource limits

### Compliance Framework
- Comprehensive rule engine for business constraints
- Automatic content filtering and validation
- Audit trail and violation tracking

### Memory Privacy
- PII detection and filtering
- Secret leakage prevention
- Configurable retention policies

## Performance Considerations

### Caching
- Letta client connection pooling
- OpenAI response caching (when appropriate)
- Qdrant connection reuse

### Async Support
- Ready for async/await patterns
- Non-blocking I/O for external services
- Configurable timeout handling

### Resource Management
- Sandbox lifecycle management
- Memory cleanup and garbage collection
- Connection pooling and reuse

## Monitoring and Observability

### Metrics
- Token usage and cost tracking
- Execution time and success rates
- Error rates and failure modes

### Tracing
- Distributed tracing with OpenTelemetry
- LangSmith evaluation integration
- Custom span attributes for domain-specific metrics

### Logging
- Structured logging throughout
- Configurable log levels
- Integration with telemetry systems

## Future Enhancements

The architecture is designed to support:
- Additional LLM providers
- Extended HITL workflows
- More sophisticated compliance rules
- Advanced RAG capabilities
- Multi-agent coordination
- Workflow orchestration scaling

This comprehensive backend application provides a solid foundation for building sophisticated AI agent workflows with enterprise-grade reliability, observability, and compliance capabilities.