# Makefile — AI Agents demo
# Usage: `make help`

SHELL := /bin/bash

# ---- Config ----
ENV_NAME ?= ai-agents-demo
API_HOST ?= 127.0.0.1
API_PORT ?= 8000
APP_MODULE ?= src.backend.main:app          # change if your FastAPI entry differs
FRONTEND_DIR ?= src.frontend
NOTEBOOKS_DIR ?= src.notebooks
CONDARUN := conda run -n $(ENV_NAME)

# ---- Meta ----
.PHONY: help env env-update kernel test run-api dev-ui dev \
        frontend-setup migrate notebooks nb1 nb2 nb3 \
        qdrant-up qdrant-down clean

help: ## Show available commands
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
	awk -F':|##' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$3}'

# ---- Environment ----
env: ## Create Conda env from environment.yml (if missing)
	@if conda env list | awk '{print $$1}' | grep -x "$(ENV_NAME)" >/dev/null 2>&1; then \
		echo "[env] Conda env '$(ENV_NAME)' already exists."; \
	else \
		echo "[env] Creating env '$(ENV_NAME)' from environment.yml"; \
		conda env create -f environment.yml; \
	fi

env-update: ## Update Conda env to match environment.yml (prunes extras)
	conda env update -f environment.yml --prune

kernel: ## Register Jupyter kernel for this environment
	$(CONDARUN) python -m ipykernel install --user --name $(ENV_NAME) --display-name "Python ($(ENV_NAME))"

# ---- Backend / Tests ----
test: ## Run Python tests (pytest -q)
	$(CONDARUN) pytest -q

run-api: ## Start FastAPI (Uvicorn) with reload
	$(CONDARUN) uvicorn $(APP_MODULE) --host $(API_HOST) --port $(API_PORT) --reload

# ---- Frontend ----
frontend-setup: ## Install UI deps (pnpm via Corepack, fallback to npm -g)
	cd $(FRONTEND_DIR) && (corepack enable && corepack prepare pnpm@9.12.2 --activate || npm i -g pnpm@9.12.2) && pnpm install

migrate: ## Run Prisma dev migration (SQLite)
	cd $(FRONTEND_DIR) && pnpm dlx prisma migrate dev --name init

dev-ui: ## Start Next.js dev server
	cd $(FRONTEND_DIR) && pnpm dev

dev: ## Start API (bg) + UI (fg). Ctrl+C to stop UI; kill API if needed.
	-$(MAKE) run-api &
	$(MAKE) dev-ui

# ---- Notebooks ----
notebooks: ## Open JupyterLab
	$(CONDARUN) jupyter lab

nb1: ## Open Notebook 1 (one-shot coding agent)
	$(CONDARUN) jupyter lab $(NOTEBOOKS_DIR)/01_one_shot_agent.ipynb

nb2: ## Open Notebook 2 (rebalancing agent + tools)
	$(CONDARUN) jupyter lab $(NOTEBOOKS_DIR)/02_rebalancing_agent.ipynb

nb3: ## Open Notebook 3 (graphs + memory + compliance)
	$(CONDARUN) jupyter lab $(NOTEBOOKS_DIR)/03_graph_memory_compliance.ipynb

# ---- Qdrant (optional, via Docker) ----
qdrant-up: ## Run Qdrant locally on :6333 (Docker required)
	docker run --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage -d qdrant/qdrant:v1.11.0

qdrant-down: ## Stop & remove local Qdrant container
	-docker rm -f qdrant

# ---- Cleanup ----
clean: ## Remove caches & pyc
	find . -type d -name "__pycache__" -exec rm -rf {} +; \
	find . -type f -name "*.pyc" -delete
