# NB1 E2B Coding Agent Documentation

## Overview

**NB1_E2B_coding_agent_v2.ipynb** is a comprehensive Jupyter notebook that demonstrates the implementation of an AI-powered coding agent using E2B (secure sandboxing), OpenAI's language models, LangGraph for workflow orchestration, and OpenTelemetry for observability. The notebook showcases both simple one-shot coding agents and advanced Human-in-the-Loop (HITL) workflows with complete tracing and artifact management.

## Main Purpose

The notebook serves multiple purposes:

1. **Demonstrate secure code execution** using E2B sandboxes (Firecracker microVMs)
2. **Showcase AI coding agents** that can generate, execute, and self-heal Python code
3. **Implement Human-in-the-Loop workflows** with LangGraph state management
4. **Provide comprehensive observability** using OpenTelemetry and LangSmith
5. **Enable artifact management** through tar archive generation and loading

## Key Technologies and Libraries

### Core Technologies
- **E2B Code Interpreter**: Secure sandboxed code execution in Firecracker microVMs
- **OpenAI Responses API**: Code generation using GPT models
- **LangGraph**: State management and workflow orchestration with interrupts
- **OpenTelemetry**: Distributed tracing with GenAI semantic conventions
- **LangSmith**: Trace visualization and evaluation platform

### Python Libraries
```python
# Core dependencies
openai==1.108.1          # OpenAI API client
e2b-code-interpreter     # E2B sandbox SDK
langgraph               # Workflow orchestration
langsmith               # Tracing and evals
opentelemetry          # Observability framework
pydantic>=2.7          # Data validation and parsing
tenacity>=8.2          # Retry mechanisms
python-dotenv>=1.0     # Environment management
```

### Supporting Libraries
- `pathlib`, `os`, `json` - File and path management
- `tarfile`, `io` - Archive creation and manipulation
- `time`, `uuid` - Timing and unique identifiers
- `typing`, `dataclasses`, `enum` - Type annotations and data structures

## Architecture Overview

The notebook implements a layered architecture:

```
┌─────────────────────────┐
│   Human Interface      │  (Jupyter cells, approval UI)
├─────────────────────────┤
│   LangGraph Workflow   │  (State management, interrupts)
├─────────────────────────┤
│   OpenTelemetry Layer  │  (Tracing, spans, metrics)
├─────────────────────────┤
│   Agent Logic Layer    │  (Code generation, execution)
├─────────────────────────┤
│   E2B Sandbox Layer    │  (Secure code execution)
└─────────────────────────┘
```

## Detailed Component Breakdown

### 1. Environment Setup and Configuration

**Purpose**: Configure API keys, models, and environment variables.

**Key Components**:
- Environment variable validation for `OPENAI_API_KEY` and `E2B_API_KEY`
- Model selection with `NB1_OPENAI_MODEL` (defaults to `gpt-5-nano`)
- Optional OpenTelemetry and LangSmith configuration

```python
# Required environment variables
OPENAI_API_KEY    # OpenAI API access
E2B_API_KEY       # E2B sandbox access

# Optional environment variables
NB1_OPENAI_MODEL           # Model selection (default: gpt-5-nano)
OPENAI_PROJECT_ID          # OpenAI project organization
LANGSMITH_TRACING          # Enable LangSmith tracing
LANGSMITH_API_KEY          # LangSmith API access
OTEL_EXPORTER_OTLP_ENDPOINT # OpenTelemetry endpoint
```

### 2. Core Agent Functions

#### Code Generation (`llm_generate_code`)
**Purpose**: Generate Python code using OpenAI's Responses API.

**Implementation**:
```python
def llm_generate_code(task: str) -> str:
    """Ask the model to produce a single Python script as a fenced block."""
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
    )
    return extract_code_blocks(resp.output[0].content[0].text)
```

**Key Features**:
- Uses strict system prompt for disciplined code generation
- Extracts code from fenced blocks using regex
- Handles API response variations with fallbacks

#### Code Execution (`run_in_e2b`)
**Purpose**: Execute generated code in secure E2B sandbox.

**Implementation**:
```python
def run_in_e2b(code: str) -> dict:
    """Create a temporary sandbox, run the code, return logs."""
    with Sandbox.create() as sbx:
        exec_result = sbx.run_code(code)
        return summarize_execution(exec_result)
```

**Key Features**:
- Automatic sandbox lifecycle management
- Comprehensive execution result parsing
- Support for stdout, stderr, and text outputs

#### Simple Coding Agent (`coding_agent`)
**Purpose**: One-shot coding agent with self-healing retry mechanism.

**Implementation**:
```python
def coding_agent(task: str, max_retries: int = 1) -> dict:
    code = llm_generate_code(task)
    first = run_in_e2b(code)
    attempt = 0
    while failed(first) and attempt < max_retries:
        # Generate feedback and retry with error context
        feedback = create_error_feedback(first)
        code = llm_generate_code(task + "\n\n" + feedback)
        first = run_in_e2b(code)
        attempt += 1
    return {"code": code, "execution": first, "retries": attempt}
```

**Key Features**:
- Single retry with error feedback
- Failure detection based on stderr and error keywords
- Complete execution history tracking

### 3. Persistent Sandbox Management

#### Sandbox Lifecycle
**Purpose**: Manage long-lived sandboxes for complex workflows.

**Key Functions**:
- `new_persistent_sandbox()` - Create sandbox with extended timeout
- `list_running_or_paused()` - Monitor active sandboxes
- `kill_by_id()` / `kill_all_running()` - Cleanup utilities

**Implementation Details**:
```python
def new_persistent_sandbox(timeout_s: int = PERSIST_TIMEOUT_SECONDS):
    """Create a persistent sandbox with extended lifetime."""
    return Sandbox.create(timeout=timeout_s)
```

### 4. Artifact Management System

#### Tar Archive Generation
**Purpose**: Package project files for download and distribution.

**Key Features**:
- Automatic project structure creation
- Code and documentation packaging
- Base64 encoding for notebook display
- Tar archive compression

**Implementation Flow**:
1. Create project directory structure
2. Write generated code and documentation
3. Create tar archive with proper metadata
4. Encode for transport/display

#### File Operations
```python
def create_tar_in_sandbox(sandbox, project_dir: str, output_path: str) -> dict:
    """Create tar archive of project directory in E2B sandbox."""
    # Implementation handles file packaging and compression
```

### 5. Human-in-the-Loop (HITL) Workflow

#### State Management
**Purpose**: Comprehensive state tracking for interactive workflows.

**State Schema**:
```python
class HITLState(TypedDict):
    user_id: str
    thread_id: str
    user_query: str
    generated_code: str
    execution_result: Dict[str, Any]
    final_result: str

    # HITL-specific fields
    stage: HITLStage
    approval_payload: Optional[ApprovalPayload]
    human_decision: Optional[DecisionPayload]
    total_cost: float
    error_message: Optional[str]
```

#### Workflow Nodes

##### 1. Code Generation Node (`generate_code_node`)
**Purpose**: Generate code with full tracing and cost tracking.

**Key Features**:
- OpenTelemetry span creation with attributes
- Token and cost calculation
- Error handling with span status updates
- LangSmith trace integration

##### 2. Code Review Node (`code_review_node`)
**Purpose**: Interrupt workflow for human code review.

**Implementation**:
```python
def code_review_node(state: HITLState) -> HITLState:
    """Handle code review stage - interrupts for human input."""
    return Command(
        update={"stage": HITLStage.CODE_REVIEW},
        goto=interrupt("Please review the generated code")
    )
```

##### 3. Code Execution Node (`execute_code_node`)
**Purpose**: Execute approved code with comprehensive tracing.

**Key Features**:
- Persistent sandbox utilization
- Fallback to temporary sandbox
- Execution metrics and latency tracking
- Comprehensive error handling

##### 4. Artifact Saving Node (`save_artifacts_node`)
**Purpose**: Package and save project artifacts.

**Key Features**:
- Project structure creation
- Tar archive generation
- Metadata preservation
- Download preparation

#### Workflow Orchestration
**Purpose**: Define the complete HITL workflow with proper routing.

**Graph Structure**:
```
START → generate_code → code_review → [INTERRUPT]
         ↓ (human approval)
      execute_code → save_artifacts → END
```

**Routing Logic**:
- Human approval → Execute code
- Human edit → Regenerate with modifications
- Human rejection → End workflow

### 6. OpenTelemetry Integration

#### Trace Configuration
**Purpose**: Comprehensive observability with GenAI semantic conventions.

**Key Components**:
- TracerProvider with OTLP export
- Console fallback for development
- GenAI semantic attributes
- LangSmith integration

**Span Attributes**:
```python
# Node-level attributes
span.set_attribute("node", "generate_code")
span.set_attribute("user_query", user_query)

# GenAI semantic conventions
span.set_attribute("gen_ai.system", "openai")
span.set_attribute("gen_ai.request.model", MODEL)
span.set_attribute("gen_ai.usage.input_tokens", tokens_in)
span.set_attribute("gen_ai.usage.output_tokens", tokens_out)
```

#### LangSmith Integration
**Purpose**: Trace visualization and evaluation platform integration.

**Implementation**:
```python
@traceable(name="generate_code_node")
def generate_code_node(state: HITLState) -> HITLState:
    # Function automatically traced in LangSmith
```

### 7. Self-Contained HITL Demonstration

#### Simplified Workflow
**Purpose**: Demonstrate core HITL concepts without full graph complexity.

**Key Features**:
- Direct function calls instead of graph execution
- Simplified state management
- Focus on artifact generation
- Educational demonstration

## Data Flows and Interactions

### 1. Simple Coding Agent Flow
```
User Query → Code Generation → E2B Execution → [Success/Failure]
                ↓ (if failure)
            Error Feedback → Retry Generation → E2B Execution
```

### 2. HITL Workflow Flow
```
User Query → Code Generation → Human Review → [Approve/Edit/Reject]
                                    ↓ (approve)
              Artifact Creation ← E2B Execution ← Code Execution
                    ↓
              Tar Archive → Download/Display
```

### 3. Tracing Data Flow
```
User Action → OTEL Span Creation → Attribute Collection → Export
                    ↓                      ↓
              LangSmith Trace         Console Output
```

## Setup Requirements and Dependencies

### Environment Prerequisites
1. **Python Environment**: Python 3.12+ with conda/pip
2. **API Keys**: OpenAI and E2B account credentials
3. **Optional Services**: LangSmith account, OTEL collector

### Installation Steps
```bash
# Core packages
pip install e2b-code-interpreter openai==1.108.1 python-dotenv>=1.0

# LangGraph and observability
pip install langgraph langsmith opentelemetry-api opentelemetry-sdk

# Supporting packages
pip install tenacity>=8.2 pydantic>=2.7
```

### Environment Configuration
```bash
# Required
export OPENAI_API_KEY="sk-..."
export E2B_API_KEY="e2b_..."

# Optional
export NB1_OPENAI_MODEL="gpt-4"
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="ls_..."
export LANGSMITH_PROJECT="nb1-coding-agent"
```

## Expected Inputs and Outputs

### Inputs

#### Simple Coding Agent
- **Input**: Natural language task description
- **Example**: "Write a Python function to calculate fibonacci numbers"

#### HITL Workflow
- **Input**: Complex coding task requiring human oversight
- **Example**: "Create a data processing class with CSV loading, cleaning, and statistical analysis"

### Outputs

#### Simple Coding Agent Output
```python
{
    "code": "def fibonacci(n):\n    # Generated code...",
    "execution": {
        "text": "Test passed successfully",
        "stdout": ["Fibonacci sequence: 0, 1, 1, 2, 3, 5"],
        "stderr": []
    },
    "retries": 0
}
```

#### HITL Workflow Output
- **Immediate**: Interactive code review interface
- **Final**: Complete project artifacts in tar archive
- **Traces**: Full execution trace in LangSmith/OTEL

#### Artifact Structure
```
project_name/
├── main.py              # Generated code
├── README.md            # Project documentation
├── requirements.txt     # Dependencies
└── metadata.json        # Execution metadata
```

## Error Handling and Edge Cases

### 1. API Failures
**Error Types**: OpenAI API errors, rate limiting, authentication
**Handling**: Exponential backoff, graceful degradation, informative error messages

### 2. Sandbox Issues
**Error Types**: E2B connection failures, timeout, resource limits
**Handling**: Automatic fallback, sandbox cleanup, resource monitoring

### 3. Code Execution Failures
**Error Types**: Runtime errors, syntax errors, infinite loops
**Handling**: Timeout protection, error feedback for retry, safe isolation

### 4. Human Input Validation
**Error Types**: Invalid decisions, malformed code edits
**Handling**: Input validation, sanitization, default fallbacks

### 5. Tracing Infrastructure
**Error Types**: OTEL export failures, LangSmith connectivity
**Handling**: Console fallback, graceful degradation, error logging

## Integration with Other Systems

### 1. LangSmith Platform
- **Purpose**: Trace visualization and evaluation
- **Integration**: Automatic trace export with `@traceable` decorator
- **Benefits**: Performance monitoring, debugging, evaluation datasets

### 2. OpenTelemetry Ecosystem
- **Purpose**: Observability and monitoring
- **Integration**: Standard OTEL spans and metrics
- **Benefits**: Integration with monitoring platforms (Jaeger, Zipkin, etc.)

### 3. E2B Platform
- **Purpose**: Secure code execution
- **Integration**: Direct SDK integration
- **Benefits**: Isolation, security, scalability

### 4. File System Integration
- **Purpose**: Artifact management and persistence
- **Integration**: Tar archives, local file operations
- **Benefits**: Portability, version control integration

## Performance Considerations

### 1. Sandbox Lifecycle
- **Optimization**: Reuse persistent sandboxes for multiple operations
- **Tradeoff**: Memory usage vs. startup latency
- **Best Practice**: Implement proper cleanup and timeout handling

### 2. Token Usage
- **Monitoring**: Track input/output tokens for cost management
- **Optimization**: Use appropriate model sizes for tasks
- **Best Practice**: Implement token budgets and limits

### 3. Trace Data Volume
- **Consideration**: High-frequency tracing can generate large data volumes
- **Mitigation**: Sampling, filtering, batch processing
- **Best Practice**: Configure appropriate trace retention policies

## Security Considerations

### 1. Code Execution Security
- **Risk**: Arbitrary code execution
- **Mitigation**: E2B sandbox isolation, timeout limits
- **Best Practice**: Never execute untrusted code outside sandboxes

### 2. API Key Management
- **Risk**: Credential exposure
- **Mitigation**: Environment variables, secure storage
- **Best Practice**: Rotate keys regularly, use least privilege access

### 3. Data Privacy
- **Risk**: Sensitive data in traces/logs
- **Mitigation**: Data sanitization, secure trace storage
- **Best Practice**: Implement data classification and retention policies

## Usage Examples

### Example 1: Simple Code Generation
```python
# Generate and execute a simple utility function
result = coding_agent("Write a function to validate email addresses using regex")
print(f"Generated code: {result['code']}")
print(f"Execution result: {result['execution']}")
```

### Example 2: HITL Workflow
```python
# Start interactive workflow
thread_id, state = start_hitl_workflow(
    "Create a web scraper class with rate limiting and error handling"
)

# Display approval interface
display_approval_request(state)

# Continue after human approval
final_state = continue_hitl_workflow(thread_id, approved_decision)
```

### Example 3: Artifact Creation
```python
# Create project with generated code
workflow = SelfContainedHITLWorkflow()
result = workflow.run_complete_workflow(
    "Build a command-line calculator with history"
)
print(f"Artifact saved to: {result['tar_path']}")
```

## Best Practices

### 1. Error Handling
- Always implement comprehensive error handling
- Provide meaningful error messages for debugging
- Use appropriate logging levels

### 2. Resource Management
- Clean up sandboxes after use
- Monitor token usage and costs
- Implement appropriate timeouts

### 3. Observability
- Use consistent naming for spans and attributes
- Include relevant context in traces
- Monitor system performance metrics

### 4. Human Interaction
- Provide clear interfaces for human input
- Validate and sanitize user inputs
- Implement appropriate feedback mechanisms

### 5. Security
- Never expose API keys in code or logs
- Validate all external inputs
- Use secure communication channels

## Troubleshooting Guide

### Common Issues

#### 1. Sandbox Connection Failures
**Symptoms**: Connection timeout, authentication errors
**Solutions**: Check E2B API key, verify network connectivity, check account limits

#### 2. OpenAI API Errors
**Symptoms**: Rate limiting, authentication failures
**Solutions**: Verify API key, implement backoff, check usage limits

#### 3. Missing Dependencies
**Symptoms**: Import errors, module not found
**Solutions**: Install required packages, check Python environment

#### 4. Trace Export Failures
**Symptoms**: Missing traces in LangSmith/OTEL
**Solutions**: Check configuration, verify connectivity, enable console fallback

#### 5. Artifact Generation Issues
**Symptoms**: Empty tar files, missing project structure
**Solutions**: Check sandbox file operations, verify permissions, debug tar creation

### Debug Commands
```python
# Check sandbox status
list_running_or_paused()

# Test API connectivity
client.models.list()

# Verify environment variables
print({k: v for k, v in os.environ.items() if 'API_KEY' in k})

# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

The NB1 E2B Coding Agent notebook provides a comprehensive demonstration of modern AI agent architecture, combining secure code execution, sophisticated workflow management, and enterprise-grade observability. It serves as both an educational resource and a production-ready foundation for building AI coding assistants with human oversight capabilities.

The notebook's modular design allows for easy extension and customization, while its comprehensive error handling and observability features ensure robust operation in production environments. The integration of multiple cutting-edge technologies demonstrates best practices for building scalable, secure, and maintainable AI agent systems.