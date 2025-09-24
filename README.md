# 🤖 AI Agents Demo — Complete Agent Ecosystem

> **Production-Ready AI Agents** showcasing LangGraph • OpenAI • Letta • Qdrant • OTEL • LangSmith • E2B • MCP • Next.js integration patterns with Human-in-the-Loop workflows, comprehensive observability, and enterprise-grade compliance frameworks.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6.7-green.svg)](https://langchain-ai.github.io/langgraph/)
[![OpenAI](https://img.shields.io/badge/OpenAI-1.108.1-orange.svg)](https://platform.openai.com/docs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 🎯 Project Overview

This repository demonstrates **end-to-end AI agent workflows** combining the latest technologies in agent orchestration, secure code execution, persistent memory, and comprehensive observability. Built for the **WBS Conference in Palermo**, it showcases production-ready patterns for building sophisticated AI systems.

### 🚀 What You'll Learn

- **LangGraph Orchestration**: State-driven workflows with human-in-the-loop interrupts
- **Secure Code Execution**: E2B sandboxes with Firecracker microVMs
- **Persistent Memory**: Letta integration for cross-session context
- **RAG Implementation**: Qdrant vector search with compliance rules
- **Comprehensive Observability**: OpenTelemetry + LangSmith tracing
- **Production API Design**: FastAPI backend with interactive Jupyter notebooks

## 🏗️ System Architecture

The project implements a **layered, production-ready architecture**:

```
                    Current System Architecture

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Notebook Layer  │    │   API Layer     │    │  Agent Layer    │
│                 │    │  (FastAPI)      │    │                 │
│ • Jupyter NB1   │───▶│ • /invoke       │───▶│ • LangGraph     │
│ • Jupyter NB2   │    │ • /resume       │    │ • HITL Gates    │
│ • Jupyter NB3   │    │ • /healthz      │    │ • State Mgmt    │
│ • HITL Demo     │    │ • /debug        │    │ • Workflows     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
        ┌─────────────────┐                   ┌─────────────────┐
        │ Processing Layer│◀─────────────────▶│ Observability   │
        │                 │                   │                 │
        │ • OpenAI Models │                   │ • OpenTelemetry │
        │ • E2B Sandboxes │                   │ • LangSmith     │
        │ • Letta Memory  │                   │ • Trace Data    │
        │ • Qdrant Vector │                   │ • Analytics     │
        └─────────────────┘                   └─────────────────┘
```

## 📁 Repository Structure

```
WBS-conference-palermo/
├── 📚 src/
│   ├── backend/                    # FastAPI + LangGraph backend
│   │   └── app/
│   │       ├── state.py           # Pydantic v2 state models
│   │       ├── graph.py           # LangGraph workflow definitions
│   │       ├── nodes.py           # Core workflow nodes
│   │       ├── hitl_nodes.py      # Human-in-the-loop nodes
│   │       ├── memory.py          # Letta memory integration
│   │       ├── compliance.py      # Compliance framework
│   │       ├── agents/            # Agent implementations
│   │       ├── e2b/              # E2B sandbox integration
│   │       ├── llm/              # LLM clients and utilities
│   │       ├── telemetry/        # OpenTelemetry setup
│   │       └── tools/            # External integrations
│   └── notebooks/                  # Demo Jupyter notebooks
│       ├── NB1_E2B_coding_agent.ipynb
│       ├── NB2_RAGwLetta.ipynb
│       ├── NB3_Comprehensive_Multiagent_System.ipynb
│       └── HITL_Calculator_Demo.ipynb
├── 📄 docs/                       # Comprehensive documentation
│   ├── NB1-E2B-coding-agent-documentation.md
│   ├── backend-app-documentation.md
│   └── NB2-RAG-Letta-documentation.md
├── ⚙️ environment.yaml            # Conda environment
├── 🔧 Makefile                    # Build and dev commands
├── 📋 CLAUDE.md                   # Claude Code instructions
├── 📖 HITL_USAGE_GUIDE.md         # HITL workflow guide
└── 🔗 INTEGRATION.md              # Backend integration patterns
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** with Conda
- **Docker** (for Qdrant and services)
- **API Keys**: OpenAI, E2B, LangSmith

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/WBS-conference-palermo
cd WBS-conference-palermo

# Create and activate conda environment
make env
conda activate ai-agents-demo

# Register Jupyter kernel
make kernel
```

### 2. Configuration

```bash
# Copy environment template
cp src/infra/.env.example src/infra/.env

# Edit with your API keys
nano src/infra/.env
```

Required environment variables:

```env
# Core APIs
OPENAI_API_KEY=sk-your-openai-key
E2B_API_KEY=your-e2b-key

# Tracing & Observability
LANGSMITH_API_KEY=ls-your-langsmith-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ai-agents-demo

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=optional-for-cloud

# Memory System
LETTA_BASE_URL=http://localhost:8283
LETTA_API_KEY=optional
```

### 3. Start Services

```bash
# Start vector database
make qdrant-up

# Start FastAPI backend
make run-api
```

### 4. Run Demo Notebooks

```bash
# Open individual notebooks
make nb1    # E2B coding agent with HITL
make nb2    # RAG + Letta memory integration
make nb3    # Comprehensive multi-agent system

# Or open JupyterLab
make notebooks
```

## 🔬 Demo Notebooks

### NB1: E2B Coding Agent with HITL

**Purpose**: Secure AI code generation with human oversight

```
User Request → Code Generation → Human Review → Decision
                     ↑                             │
                     │                             ▼
               ┌─────────┐                 ┌───────────────┐
               │  Edit   │                 │   Approve?    │
               │ (retry) │                 │               │
               └─────────┘                 │ ✓ Execute     │
                     ▲                     │ ✗ Reject      │
                     └─────────────────────┴───────────────┘
                                                   │
                                                   ▼
                                        E2B Execution → Artifacts
```

**Key Features**:

- ✅ OpenAI Responses API for code generation
- ✅ E2B Firecracker sandboxes for secure execution
- ✅ Human-in-the-loop approval workflows
- ✅ Comprehensive OpenTelemetry tracing
- ✅ Artifact management with TAR archives

**Demo Flow**:

1. Generate Python code from natural language
2. Human reviews and approves/edits code
3. Execute in isolated E2B sandbox
4. Collect artifacts (files, charts, logs)
5. Full tracing in LangSmith dashboard

### NB2: RAG with Letta Memory

**Purpose**: Intelligent document retrieval with persistent memory

```
                    NB2: RAG + Memory Workflow

    User Query
        │
        ▼
┌──────────────────┐    ┌─────────────────┐
│ Memory Recall    │◀──▶│ Letta Memory    │
│ (Past Context)   │    │ API             │
└──────────────────┘    └─────────────────┘
        │
        ▼
┌──────────────────┐    ┌─────────────────┐
│ RAG Search       │◀──▶│ Qdrant Vector   │
│ (Rules & Docs)   │    │ Database        │
└──────────────────┘    └─────────────────┘
        │
        ▼
┌──────────────────┐
│ Context Assembly │
│ (Memory + RAG)   │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ LLM Analysis     │
│ (Generate Reply) │
└──────────────────┘
        │
        ▼
┌──────────────────┐    ┌─────────────────┐
│ Memory Update    │───▶│ Save Session    │
│ (Save Results)   │    │ for Future      │
└──────────────────┘    └─────────────────┘
        │
        ▼
    Response
```

**Key Features**:

- ✅ Qdrant vector search with semantic similarity
- ✅ Letta persistent memory across sessions
- ✅ LangGraph state management and checkpointing
- ✅ Compliance rules retrieval and analysis
- ✅ Portfolio rebalancing use case

**Demo Scenarios**:

1. "What are sector concentration limits?" → RAG retrieval
2. "Following up on tech exposure..." → Memory context
3. "Credit rating requirements?" → Multi-rule synthesis
4. Session persistence across notebook restarts

### NB3: Comprehensive Multi-Agent System

**Purpose**: Production-grade agent coordination with compliance

```
                NB3: Multi-Agent System with Compliance

    Task Planning
         │
         ▼
  Agent Coordination ──────┐
         │                 │
         ▼                 │
  ┌──────────────────┐     │     ┌──────────────────┐
  │ Parallel Agent   │     │     │ Compliance Gate  │
  │ Execution        │     └────▶│ Rule Checking    │
  │ • Research       │           │                  │
  │ • Analysis       │           └──────────────────┘
  │ • Calculation    │                    │
  └──────────────────┘                    ▼
         │                         ┌─────────────┐
         │                         │ Compliant?  │
         │                         │             │
         │                         │ ✓ Continue  │
         │                         │ ✗ Handle    │
         │                         └─────────────┘
         │                              │     │
         ▼                              │     ▼
  Results Synthesis ◀──────────────────┘  Human Review
         │                                     │
         ▼                                     │
  Memory Persistence ◀─────────────────────────┘
```

**Key Features**:

- ✅ Multi-agent coordination patterns
- ✅ Compliance framework with rule engine
- ✅ Advanced HITL workflows with conditional routing
- ✅ Integration with external tools (Exa, calculators)
- ✅ End-to-end observability and evaluation

### HITL Calculator Demo

**Purpose**: Interactive calculator with human verification

**Key Features**:

- ✅ Simple HITL workflow demonstration
- ✅ Real-time human input collection
- ✅ State persistence and resumption
- ✅ Error handling and recovery patterns

## 🔧 Development Commands

### Core Development

```bash
# Environment management
make env              # Create conda environment
make env-update       # Update environment packages
make kernel           # Register Jupyter kernel

# Service management
make qdrant-up        # Start Qdrant vector database
make qdrant-down      # Stop Qdrant
make run-api          # Start FastAPI backend
make dev-ui           # Start Next.js frontend
make dev              # Start both API and UI

# Testing and quality
make test             # Run pytest suite
make lint             # Code linting and formatting
make clean            # Clean cache files
```

### API Development

```bash
make run-api          # Start FastAPI backend
make test             # Run test suite
make debug/env        # Check environment setup
```

### Notebooks

```bash
make notebooks        # Open JupyterLab
make nb1             # Open NB1 (E2B coding agent)
make nb2             # Open NB2 (RAG + Letta)
make nb3             # Open NB3 (Multi-agent system)
```

## 🧪 Key Technologies

### Agent Orchestration

- **[LangGraph 0.6.7](https://langchain-ai.github.io/langgraph/)**: State-driven workflow orchestration
- **[Pydantic v2](https://docs.pydantic.dev/latest/)**: Type-safe data validation and serialization
- **[OpenAI 1.108.1](https://platform.openai.com/docs)**: Advanced language models with structured outputs

### Secure Execution

- **[E2B Sandboxes](https://e2b.dev/docs)**: Firecracker microVMs for code execution
- **Artifact Management**: TAR archives, file system integration
- **Timeout Protection**: Configurable execution limits

### Memory & Retrieval

- **[Letta 0.11.7](https://docs.letta.com/)**: Persistent agent memory with HTTP API
- **[Qdrant 1.15.1](https://qdrant.tech/documentation/)**: Vector database for semantic search
- **Embedding Models**: OpenAI text-embedding-ada-002

### Observability

- **[OpenTelemetry 1.37.0](https://opentelemetry.io/docs/languages/python/)**: Distributed tracing
- **[LangSmith 0.4.29](https://docs.langchain.com/langsmith)**: LLM monitoring and evaluation
- **GenAI Semantic Conventions**: Standardized trace attributes

### Web Framework

- **[FastAPI 0.115.5](https://fastapi.tiangolo.com/)**: High-performance Python API
- **[Jupyter](https://jupyter.org/)**: Interactive notebook development environment
- **[Uvicorn](https://www.uvicorn.org/)**: ASGI server for FastAPI

## 📊 HITL Workflow Examples

### API Usage

**Start HITL Session**:

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "user_query": "Create a Python function to analyze portfolio risk",
    "thread_id": "session_001"
  }'
```

**Response** (202 Accepted):

```json
{
  "status": "interrupted",
  "thread_id": "session_001",
  "stage": "code_review",
  "approval_payload": {
    "code": "def analyze_portfolio_risk(positions, market_data):\n    # Generated risk analysis code...",
    "task": "Create a Python function to analyze portfolio risk",
    "suggestion": "Please review the code and choose: approve, edit, or reject"
  }
}
```

**Resume with Decision**:

```bash
curl -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "session_001",
    "decision": {
      "decision": "approve"
    }
  }'
```

**Final Response** (200 OK):

```json
{
  "status": "completed",
  "result": "Portfolio risk analysis completed successfully",
  "sandbox_execution": {
    "stdout": ["Risk metrics calculated", "Charts saved to /outputs"],
    "success": true
  },
  "total_cost": 0.0089,
  "artifacts_path": "portfolio_analysis_artifacts.tar.gz"
}
```

### Workflow States

```
                        HITL Workflow State Machine

         START
           │
           ▼
    ┌─────────────┐
    │  Planning   │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │    Code     │
    │ Generation  │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │   Human     │──── approve ────┐
    │   Review    │                 │
    └─────────────┘                 ▼
           │                 ┌─────────────┐
        edit│                │  Execution  │
           │                 └─────────────┘
           ▼                        │
    ┌─────────────┐            success│  │error
    │   Retry     │                 │  │
    │ Generation  │◀────────────────┘  │
    └─────────────┘                    │
           │                           ▼
           └───────────────────► ┌─────────────┐
                                │   SUCCESS   │
         reject                 │     END     │
           │                    └─────────────┘
           ▼
    ┌─────────────┐
    │  REJECTED   │
    │     END     │
    └─────────────┘
```

## 🔒 Security & Compliance

### Sandbox Security

- **Isolated Execution**: Firecracker microVMs with no network access
- **Resource Limits**: CPU, memory, and time constraints
- **File System Isolation**: Separate file systems per sandbox

### Compliance Framework

- **Model Controls**: Approved model lists, token budgets
- **Tool Safety**: Domain restrictions, command filtering
- **Memory Privacy**: PII detection, secret sanitization
- **Output Quality**: Content validation, citation requirements

### API Security

- **Environment Variables**: Secure credential management
- **HTTPS Endpoints**: TLS encryption for all communications
- **Input Validation**: Pydantic models for request validation

## 📈 Performance & Monitoring

### Observability Dashboard

The system provides comprehensive monitoring through multiple channels:

**OpenTelemetry Traces**:

- Node-level execution tracking
- LLM token usage and costs
- Human decision timing
- Error rates and recovery

**LangSmith Integration**:

- LLM input/output monitoring
- Evaluation datasets and A/B testing
- Cost analysis and optimization
- Response quality metrics

**Business Metrics**:

- Success rates by workflow type
- Average human approval time
- Most common rejection reasons
- Resource utilization patterns

### Monitoring Capabilities

The system provides comprehensive monitoring through multiple channels:

**Real-time Monitoring**:

- Individual workflow execution tracking
- Resource usage monitoring (CPU, memory, tokens)
- Error rates and failure analysis
- Human decision timing and patterns

**Historical Analytics**:

- Workflow success patterns over time
- Cost analysis and optimization opportunities
- User interaction patterns and preferences
- System performance trends

## 🔮 Future Enhancements

### Planned Frontend Development
- **Next.js 15 Web UI**: Interactive dashboard for workflow management
- **Real-time Monitoring**: Live agent execution visualization
- **User Management**: Authentication and session handling
- **Workflow Builder**: Visual LangGraph workflow designer
- **Analytics Dashboard**: Performance metrics and usage analytics

### Additional Features
- **Multi-tenancy**: Support for multiple organizations
- **Advanced RAG**: Hybrid search with keyword + semantic
- **Model Router**: Dynamic model selection based on task complexity
- **Workflow Templates**: Pre-built agent patterns for common use cases
- **Enterprise SSO**: SAML/OAuth integration for enterprise deployment

## 🚀 Production Deployment

### Docker Deployment

```yaml
# docker-compose.yml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - LETTA_BASE_URL=http://letta:8283
    depends_on:
      - qdrant
      - letta

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  letta:
    image: letta/letta:latest
    ports:
      - "8283:8283"
    volumes:
      - letta_data:/app/data

volumes:
  qdrant_data:
  letta_data:
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-agents-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-agents-api
  template:
    spec:
      containers:
        - name: api
          image: ai-agents:latest
          ports:
            - containerPort: 8000
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: api-keys
                  key: openai
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `make env`
4. Run tests: `make test`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open Pull Request

### Code Standards

- **Type Hints**: All functions must have complete type annotations
- **Pydantic Models**: Use Pydantic v2 for data validation
- **Error Handling**: Comprehensive exception handling with proper logging
- **Testing**: Unit tests for all new functionality
- **Documentation**: Docstrings and README updates

## 🆘 Troubleshooting

### Common Issues

**Environment Setup**:

```bash
# Conda environment conflicts
conda clean --all
make env

# Jupyter kernel not found
make kernel

# Permission issues
chmod +x scripts/*.sh
```

**Service Connectivity**:

```bash
# Check Qdrant status
curl http://localhost:6333/health

# Check Letta connectivity
curl http://localhost:8283/health

# Verify API endpoints
curl http://localhost:8000/healthz
```

**API Key Issues**:

```bash
# Verify environment variables
make debug/env

# Test OpenAI connectivity
python -c "from openai import OpenAI; OpenAI().models.list()"

# Test E2B connection
e2b auth whoami
```

### Debug Commands

```bash
# Environment diagnostics
make debug/env          # Check environment variables
make debug/letta        # Test Letta connectivity
make debug/qdrant       # Test Qdrant connection

# Service logs
docker logs qdrant      # Qdrant container logs
docker logs letta       # Letta container logs

# API debugging
curl http://localhost:8000/debug/env    # Environment check
curl http://localhost:8000/debug/letta  # Letta status
```

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain Team** for LangGraph and LangSmith
- **E2B Team** for secure sandbox infrastructure
- **Letta Team** for persistent memory capabilities
- **Qdrant Team** for vector database technology
- **OpenTelemetry Community** for observability standards

---

## 🔗 Quick Links

- **[Live Demo](https://your-demo-url.com)** - Try the system online
- **[Documentation](./docs/)** - Detailed technical documentation
- **[API Reference](http://localhost:8000/docs)** - Interactive API docs
- **[LangSmith Dashboard](https://smith.langchain.com/)** - View traces and evaluations
- **[WBS Conference](https://wbs-conference.com)** - Conference information

**Ready to build production AI agents?** Start with `make env && make dev` and explore the notebooks! 🚀
