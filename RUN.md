# 🚀 How to Run the HITL Implementation

## Quick Start Guide

### 1. **Start the FastAPI Server**

```bash
conda run -n ai-agents-demo uvicorn src.backend.main:app --host 127.0.0.1 --port 8001 --reload
```

### 2. **Test the Server is Running**

```bash
curl http://localhost:8001/healthz
```

Expected response: `{"status":"ok"}`

### 3. **Start a HITL Coding Session**

```bash
curl -X POST http://localhost:8001/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "user_query": "Write a Python function to calculate fibonacci numbers with error handling",
    "thread_id": "demo_session"
  }'
```

Expected response: HTTP 202 with interrupt payload for human review

### 4. **Resume with Human Decision**

#### Approve the code:
```bash
curl -X POST http://localhost:8001/resume \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "demo_session",
    "decision": {"decision": "approve"}
  }'
```

#### Edit the code:
```bash
curl -X POST http://localhost:8001/resume \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "demo_session",
    "decision": {
      "decision": "approve",
      "code": "def fibonacci(n):\n    # Your edited code here\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    }
  }'
```

#### Reject the code:
```bash
curl -X POST http://localhost:8001/resume \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "demo_session",
    "decision": {"decision": "reject"}
  }'
```

### 5. **Check Session Status**

```bash
curl http://localhost:8001/threads/demo_session/status
```

## 🔧 Additional Endpoints

### Debug Environment
```bash
curl http://localhost:8001/debug/env
```

### Health Check
```bash
curl http://localhost:8001/healthz
```

### Streaming Response
```bash
curl -X POST http://localhost:8001/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "user_query": "Write a simple hello world function",
    "stream": true
  }'
```

## 📋 Workflow States

1. **Initial Request** → HTTP 202 (interrupted for human review)
2. **Human Decision** → Resume with approval/edit/reject
3. **Code Execution** → E2B sandbox runs the code
4. **Final Result** → HTTP 200 with execution results

## 🎯 Expected Flow

1. Send coding request to `/invoke`
2. Receive interrupt payload with generated code
3. Review the code and make decision
4. Send decision to `/resume`
5. Get final results with code execution output

The HITL workflow pauses for human approval and resumes based on your decisions. All interactions are traced with OpenTelemetry and sent to LangSmith for observability.