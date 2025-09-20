# HITL + OpenTelemetry Implementation Guide

## 🎯 Overview

This implementation adds Human-in-the-Loop (HITL) capabilities to the LangGraph coding agent with comprehensive OpenTelemetry tracing and LangSmith integration.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Copy and configure environment
cp src/infra/.env.example src/infra/.env

# Edit src/infra/.env with your keys:
# - OPENAI_API_KEY
# - E2B_API_KEY
# - LANGSMITH_API_KEY
```

### 2. Start the API

```bash
make run-api
```

### 3. Test the HITL Workflow

#### Initial Request (Starts HITL Session)

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "user_query": "Write a Python function that calculates the factorial of a number with error handling",
    "thread_id": "session_001"
  }'
```

**Response:** 202 Accepted (Waiting for Human Input)
```json
{
  "status": "interrupted",
  "thread_id": "session_001",
  "stage": "code_review",
  "approval_payload": {
    "code": "def factorial(n):\n    if n < 0:\n        raise ValueError...",
    "task": "Write a Python function...",
    "suggestion": "Please review the code and choose: approve, edit, or reject",
    "options": ["approve", "edit", "reject"]
  }
}
```

#### Human Decision (Resume Session)

```bash
# Approve the code
curl -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "session_001",
    "decision": {
      "decision": "approve"
    }
  }'

# OR edit the code
curl -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "session_001",
    "decision": {
      "decision": "approve",
      "code": "def factorial(n):\n    # Improved version with better error handling\n    if not isinstance(n, int):\n        raise TypeError(\"Input must be an integer\")\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    if n == 0 or n == 1:\n        return 1\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n\nif __name__ == \"__main__\":\n    print(factorial(5))\n    assert factorial(5) == 120"
    }
  }'

# OR reject the code
curl -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "session_001",
    "decision": {
      "decision": "reject",
      "reason": "Code doesn't meet requirements"
    }
  }'
```

**Final Response:** 200 OK (Completed)
```json
{
  "status": "completed",
  "thread_id": "session_001",
  "result": "Code executed successfully.\nOutput:\n120",
  "generated_code": "def factorial(n): ...",
  "sandbox_execution": {
    "stdout": ["120\n"],
    "stderr": []
  },
  "total_cost": 0.0042,
  "token_usage": {"input_tokens": 150, "output_tokens": 85}
}
```

### 4. Check Session Status

```bash
curl http://localhost:8000/threads/session_001/status
```

## 🔧 Configuration Options

### Environment Variables

```bash
# Core APIs
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-nano
E2B_API_KEY=your_key_here

# LangSmith Tracing
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=agents-demo
LANGSMITH_TRACING=true

# OTLP for LangSmith
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.smith.langchain.com/otel
OTEL_EXPORTER_OTLP_HEADERS=x-api-key=your_langsmith_key,Langsmith-Project=agents-demo
OTEL_SEMCONV_STABILITY_OPT_IN=gen-ai

# HITL Behavior
USE_CODING_WORKFLOW=true              # Enable HITL coding workflow
ENABLE_EXECUTION_REVIEW=false         # Add review after code execution
SKIP_CLEAN_EXECUTION_REVIEW=true      # Skip review if no errors
```

## 📊 OpenTelemetry + LangSmith Integration

### Traces Generated

The implementation creates comprehensive traces with GenAI semantic conventions:

1. **Root Span**: `invoke` - Overall workflow execution
2. **Node Spans**: Each LangGraph node (code_generation, code_review, execute_code)
3. **GenAI Spans**: LLM operations with standardized attributes
4. **Sandbox Spans**: E2B execution tracking

### Key Attributes

- `gen_ai.system`: "openai"
- `gen_ai.operation.name`: "responses"
- `gen_ai.request.model`: "gpt-5-nano"
- `gen_ai.prompt`: Input text (truncated)
- `gen_ai.completion`: Generated code (truncated)
- `gen_ai.usage.input_tokens`: Token counts
- `gen_ai.usage.output_tokens`: Token counts
- `hitl.stage`: Current HITL stage
- `hitl.decision`: Human decision
- `sandbox.success`: Execution result
- `sandbox.id`: E2B sandbox identifier

### View in LangSmith

1. Open [LangSmith Dashboard](https://smith.langchain.com/)
2. Navigate to your project
3. View traces with full HITL flow visualization
4. Analyze costs, latency, and decision points

## 🔀 HITL Workflow States

### State Transitions

```
START → recall → code_generation → code_review → HITL_PAUSE
                                      ↓ (human decision)
                 ┌─────────────────────┼─────────────────────┐
                 ↓ (approve)          ↓ (edit)             ↓ (reject)
           execute_code         code_generation          END
                 ↓                     ↓
          (success/retry)        code_review
                 ↓                     ↓
               save              HITL_PAUSE...
                 ↓
               END
```

### Human Decision Points

1. **Code Review**: Approve/edit/reject generated code
2. **Execution Review** (optional): Review sandbox execution results
3. **Error Recovery**: Handle execution failures with human guidance

## 🛠️ Advanced Usage

### Custom Workflow Configuration

```python
# Disable HITL for automated runs
USE_CODING_WORKFLOW=false

# Enable execution review for critical code
ENABLE_EXECUTION_REVIEW=true
SKIP_CLEAN_EXECUTION_REVIEW=false
```

### Streaming Responses

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "user_query": "Create a REST API endpoint",
    "stream": true
  }'
```

### Thread Management

```bash
# Check all active threads
curl http://localhost:8000/debug/env

# Get specific thread status
curl http://localhost:8000/threads/{thread_id}/status
```

## 🚨 Error Handling

### Common Issues

1. **Missing API Keys**: Check `.env` file configuration
2. **E2B Sandbox Timeout**: Extend timeout or check E2B status
3. **HITL Session Lost**: Thread expires (use MemorySaver persistence)
4. **Tracing Not Appearing**: Verify LangSmith API key and project name

### Debug Endpoints

```bash
# Check API status
curl http://localhost:8000/healthz

# Verify environment
curl http://localhost:8000/debug/env

# Test Letta connectivity
curl http://localhost:8000/debug/letta
```

## 🎯 Production Deployment

### Persistent Storage

For production, implement SQLite checkpointer:

```python
# Replace MemorySaver with persistent solution
from your_sqlite_implementation import SqliteSaver

checkpointer = SqliteSaver("checkpoints.db")
```

### Monitoring

- Monitor OpenTelemetry traces in LangSmith
- Track HITL decision rates and timing
- Alert on high error rates or long session times
- Monitor E2B sandbox usage and costs

### Security

- Secure API keys in environment variables
- Implement authentication for `/resume` endpoint
- Sanitize code execution in E2B sandboxes
- Audit human decisions for compliance

## 📚 API Reference

### POST /invoke
Start or continue a HITL coding workflow

### POST /resume
Resume interrupted workflow with human decision

### GET /threads/{id}/status
Check thread status and next steps

### GET /healthz
Basic health check

### GET /debug/env
Environment validation (non-sensitive data only)

---

**🎉 Ready to go!** Your HITL coding agent with full tracing is now operational.