# Makefile — common project commands.
#
# Usage:
#   make install     # one-time setup: create venv, install deps
#   make ingest      # download EU AI Act and build the vector store
#   make run         # launch the Streamlit UI
#   make graph       # run the LangGraph workflow in the terminal
#   make smoke       # smoke-test the retriever
#   make test        # run unit tests (Phase 5)
#   make clean       # remove the vector store (re-runnable from scratch)

PYTHON := python
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate

.PHONY: help install ingest run graph smoke test clean

help:
	@echo "Available targets:"
	@echo "  install   Create venv and install dependencies"
	@echo "  ingest    Download data and build the vector store"
	@echo "  run       Launch the Streamlit app"
	@echo "  graph     Run the LangGraph workflow in the terminal"
	@echo "  smoke     Smoke-test semantic retrieval"
	@echo "  test      Run unit tests"
	@echo "  clean     Remove the vector store"

install:
	python3.11 -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt
	@echo "✓ Done. Activate with: source $(VENV)/bin/activate"

ingest:
	bash scripts/download_data.sh
	$(PYTHON) -m src.ingestion

run:
	$(PYTHON) -m streamlit run app/streamlit_app.py

graph:
	$(PYTHON) -m src.graph

smoke:
	$(PYTHON) -m scripts.smoke_test_retrieval

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf data/chroma_db
	@echo "✓ Removed data/chroma_db (run 'make ingest' to rebuild)"