# 🔎 RAG Knowledge Assistant — EU AI Act

> A Retrieval-Augmented Generation (RAG) assistant that answers questions about the **EU AI Act** using a multi-step **LangGraph** workflow, local embeddings, and a swappable LLM backend (**Groq** by default, Azure OpenAI optional).

<p align="center">
  <img src="docs/architecture.png" width="720" alt="Architecture diagram"/>
</p>

---

## ✨ What this project does

Ask the assistant questions like:

- *"What are the obligations for providers of high-risk AI systems?"*
- *"Which AI practices are prohibited under the EU AI Act?"*
- *"What is the definition of a general-purpose AI model?"*

The system retrieves the most relevant passages from the EU AI Act, **grades** whether they actually answer the question, and either generates a grounded answer with citations or politely refuses if the context is insufficient.

This makes it more than a "retrieve-and-answer" demo — it's a small but real **AI system with guardrails**.

---

## 🏗️ Architecture

The workflow runs as a small **LangGraph** state machine:

1. **Rewrite** — clean up the user's question for better retrieval
2. **Retrieve** — semantic search over the EU AI Act in a Chroma vector store
3. **Grade** — does the retrieved context actually answer the question?
4. **Answer** — if yes, generate a grounded response with citations; if no, refuse cleanly

This multi-step orchestration is the key difference from a naive RAG pipeline. It reduces hallucinations and makes failures transparent and debuggable.

---

## 🛠️ Stack

| Layer | Tool |
|---|---|
| LLM | **Groq** (Llama 3.3 70B) — swappable to Azure OpenAI via env var |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector store | **Chroma** (local, persisted) |
| Orchestration | **LangChain + LangGraph** |
| Backend API | **FastAPI** |
| Frontend | **Streamlit** |
| Evaluation | Custom Q&A test set + retrieval precision + LLM-as-judge |

---

## 🚀 Quickstart

```bash
# 1. Clone and install
git clone https://github.com/hugocorreia123/rag-knowledge-assistant.git
cd rag-knowledge-assistant
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your Groq API key (free at https://console.groq.com)
cp .env.example .env
# edit .env and set GROQ_API_KEY=...

# 3. Download and ingest the EU AI Act
bash scripts/ingest.sh

# 4. Run the app
streamlit run app/streamlit_app.py
```

The Streamlit UI opens at `http://localhost:8501`.
The FastAPI endpoint runs at `http://localhost:8000/ask` (start it with `uvicorn src.api:app --reload`).

---

## 📊 Evaluation

The `evaluation/` directory contains **20 hand-written question–answer pairs** covering different chapters of the EU AI Act, plus an automated evaluation script.

Run it with:
```bash
python evaluation/evaluate.py
```

**Latest results** *(see `evaluation/results.md` for the full breakdown)*:

| Metric | Value |
|---|---|
| Retrieval precision@5 | *coming after eval run* |
| Answer faithfulness (LLM-judge) | *coming after eval run* |
| Refusal rate on out-of-scope questions | *coming after eval run* |
| Median latency | *coming after eval run* |

> Numbers will be filled in once the evaluation phase is complete.

---

## 🔄 Swapping the LLM provider

The system reads `LLM_PROVIDER` from the environment. Supported values:

- `groq` (default — free, Llama 3.3 70B)
- `azure_openai` (production — paid)
- `openai` (paid)

Switching providers is a single env variable change — no code edits.

---

## 📁 Repository layout

```
src/             # core pipeline (ingestion, retriever, graph, api)
app/             # Streamlit UI
evaluation/      # test questions + evaluation script
tests/           # unit tests
docs/            # architecture diagram, screenshots, demo GIF
scripts/         # one-command setup scripts
data/            # EU AI Act PDFs + vector store (gitignored)
```

---

## 🧠 Why this design

A few choices worth flagging for anyone reviewing the code:

- **LangGraph over plain LangChain.** The explicit state machine makes the workflow inspectable and debuggable. Each node is independently testable.
- **Local embeddings.** `all-MiniLM-L6-v2` is small (90 MB), fast on CPU, and good enough for legal text. Keeps the project provider-independent and free to run.
- **Grade-before-answer.** Most public RAG demos answer no matter what was retrieved, which is exactly how you get hallucinations. The grader node is what makes this trustworthy.
- **Evaluation set checked into the repo.** The hardest part of building RAG is *knowing* if it's good. Writing the eval first forces honesty.
- **Provider abstraction.** The same code runs on free Groq for the public demo and on enterprise Azure OpenAI in production — useful pattern, almost zero code.

---

## 📜 License

MIT — see [LICENSE](LICENSE)

---

## 👤 Author

**Hugo Correia** — Data Scientist & ML/AI Engineer
- 🔗 LinkedIn: [linkedin.com/in/hugogncorreia](https://www.linkedin.com/in/hugogncorreia)
- 💼 GitHub: [github.com/hugocorreia123](https://github.com/hugocorreia123)
- ✉️ Hugocorreia55@hotmail.com

> Built as part of a portfolio focused on production-ready data and Generative AI solutions.