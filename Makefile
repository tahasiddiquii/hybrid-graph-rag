.DEFAULT_GOAL := help
PY ?= python3.12

.PHONY: help setup install dev lint fmt test benchmark search graph mcp demo clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv (Python 3.12)
	$(PY) -m venv .venv

install: ## Install the package
	. .venv/bin/activate && pip install -e .

dev: setup ## Create venv + install with dev extras
	. .venv/bin/activate && pip install -e ".[dev]"

lint: ## Ruff lint + format check
	. .venv/bin/activate && ruff check . && ruff format --check .

fmt: ## Auto-format with ruff
	. .venv/bin/activate && ruff format . && ruff check --fix .

test: ## Run the test suite
	. .venv/bin/activate && pytest -q

benchmark: ## Run the retrieval benchmark + CI gate
	. .venv/bin/activate && hybrid-rag benchmark

search: ## Run a hybrid search (Q="...")
	. .venv/bin/activate && hybrid-rag search "$(Q)"

graph: ## Multi-hop graph retrieval (C=concept HOPS=2)
	. .venv/bin/activate && hybrid-rag graph "$(C)" --hops $(or $(HOPS),2)

mcp: ## Print the MCP tool surface + an example call
	. .venv/bin/activate && hybrid-rag mcp

demo: ## Run the end-to-end demo
	. .venv/bin/activate && hybrid-rag demo

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
