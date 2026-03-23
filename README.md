# 📄 Document RAG

A **production-grade Retrieval-Augmented Generation (RAG) system** with hybrid retrieval, cross-encoder reranking, and citation-based safety guardrails. Ask questions about your PDF documents and get factually grounded, cited answers — with automated quality evaluation built in.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Setting Up the Knowledge Base](#setting-up-the-knowledge-base)
- [Usage](#usage)
  - [Querying the RAG System](#querying-the-rag-system)
  - [Evaluating Quality](#evaluating-quality)
  - [Generating Test Data](#generating-test-data)
  - [Running Unit Tests](#running-unit-tests)
- [Evaluation & Quality Gates](#evaluation--quality-gates)
- [CI/CD Integration](#cicd-integration)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

**Document RAG** ingests PDF documents into a hybrid vector + keyword search index, retrieves the most relevant passages for any user query, and generates concise, citation-backed answers using a local LLM (via [Ollama](https://ollama.com/)). A built-in citation validator acts as a safety guardrail, ensuring every factual claim in the response maps back to a verified source in the retrieved context — blocking hallucinated or unsupported statements.

Quality is continuously monitored through [RAGAS](https://docs.ragas.io/en/stable/) metrics (faithfulness, answer relevancy, context precision, context recall) and a CI/CD pipeline that enforces quality gate thresholds before any changes are merged.

---

## Features

- **📥 PDF Ingestion** — Load and chunk PDF documents with configurable chunk size and overlap using the `tiktoken` tokenizer.
- **🔀 Hybrid Retrieval** — Combines dense vector search (Chroma + HuggingFace embeddings) with sparse BM25 keyword search in a 50/50 ensemble for robust, recall-maximizing retrieval.
- **🎯 Cross-Encoder Reranking** — A `ms-marco-MiniLM-L-6-v2` cross-encoder re-scores and re-ranks the top ensemble results to boost precision.
- **🛡️ Citation Validation** — Every LLM response is checked to confirm that all inline citations refer to actually retrieved documents. Responses citing non-existent sources are blocked.
- **🧠 Local LLM Inference** — Runs entirely on-premise using [Ollama](https://ollama.com/) with `llama3.2` (or `llama3.1`). No data is sent to external APIs by default.
- **📊 RAGAS Evaluation** — Built-in evaluation script computes faithfulness, answer relevancy, context precision, and context recall against a curated test dataset.
- **⚙️ CI/CD Quality Gate** — GitHub Actions workflow runs evaluation on every pull request and blocks merges if quality thresholds are not met.
- **🤖 Automatic Test Data Generation** — Automatically generate Q&A pairs directly from your PDF documents to bootstrap an evaluation dataset.

---

## Architecture

```
PDFs (data/)
    │
    ▼
┌─────────────────────────────────────┐
│          populate_db.py             │
│  1. Load & chunk PDFs               │
│  2. Embed chunks (all-MiniLM-L6-v2) │
│  3. Store in Chroma vector DB        │
│  4. Build & persist BM25 index       │
└────────────────┬────────────────────┘
                 │  chroma/ + bm25_index.pkl
                 ▼
┌─────────────────────────────────────┐
│           query_data.py             │
│                                     │
│  User Query                         │
│      │                              │
│      ├──► Vector Search (Chroma)    │
│      └──► BM25 Keyword Search       │
│              │                      │
│              ▼                      │
│      Ensemble Retriever (50/50)     │
│              │                      │
│              ▼                      │
│      Cross-Encoder Reranker         │  ──► Top-5 Chunks
│              │                      │
│              ▼                      │
│      LLM Generation (Ollama)        │  ──► Draft Response
│              │                      │
│              ▼                      │
│      Citation Validator             │  ──► ✅ Final Response
└─────────────────────────────────────┘         (or ❌ blocked)

Test Dataset (test.csv)
    │
    ▼
┌─────────────────────────────────────┐
│          evaluate_rag.py            │
│  RAGAS Metrics per question:        │
│  - Faithfulness                     │
│  - Answer Relevancy                 │
│  - Context Precision                │
│  - Context Recall                   │
│        │                            │
│        ▼                            │
│  Quality Gate (pass / fail)         │
│        │                            │
│        ▼                            │
│  final_evaluation_metrics.csv       │
└─────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.12 |
| **LLM Framework** | LangChain |
| **Vector Database** | Chromadb |
| **Keyword Search** | BM25 (rank-bm25) |
| **LLM Inference** | Ollama (`llama3.2`, `llama3.1`) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Reranking** | HuggingFace `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Evaluation** | RAGAS |
| **Testing** | Pytest |
| **CI/CD** | GitHub Actions |

---

## Project Structure

```
Document_RAG/
├── data/                        # Source PDF documents
│   ├── Attention_all_you_need.pdf
│   └── Baysean.pdf
├── chroma/                      # Chroma vector database (auto-generated)
├── tests/
│   └── test_rag.py              # Unit tests
├── .github/
│   └── workflows/
│       └── rag_eval.yml         # CI/CD evaluation pipeline
│
├── populate_db.py               # Ingest PDFs and build the search indexes
├── query_data.py                # Query interface with citation validation
├── evaluate_rag.py              # RAGAS-based quality evaluation
├── evaluate_rag_llama.py        # LLM-as-a-judge evaluation (alternative)
├── generate_candidates.py       # Auto-generate Q&A test datasets from PDFs
├── get_embedding_function.py    # HuggingFace embedding model loader
├── get_reranker.py              # Cross-encoder reranker loader
├── citation_validator.py        # Citation safety guardrail
├── prompts.json                 # System and evaluation prompts
│
├── bm25_index.pkl               # Serialized BM25 index (auto-generated)
├── test.csv                     # Curated test dataset (64 Q&A pairs)
├── candidate_dataset.csv        # Auto-generated candidate Q&A dataset
├── final_evaluation_metrics.csv # Latest RAGAS evaluation results
│
├── requirements.txt             # Python dependencies
└── LICENSE                      # MIT License
```

---

## Getting Started

### Prerequisites

- **Python 3.12**
- **[Ollama](https://ollama.com/)** installed and running locally
- The `llama3.2` model pulled in Ollama:
  ```bash
  ollama pull llama3.2
  ```

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VRP-github/Document_RAG.git
   cd Document_RAG
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** (create a `.env` file or export directly):
   ```bash
   export DATA_PATH="data/"
   export CHROMA_PATH="chroma/"
   ```

4. **Start Ollama** (in a separate terminal):
   ```bash
   ollama serve
   ```

### Setting Up the Knowledge Base

Place your PDF files in the `data/` directory, then run:

```bash
python populate_db.py
```

This will:
- Load all PDFs from the `data/` directory
- Split them into 500-character chunks with 100-character overlap
- Generate embeddings using `all-MiniLM-L6-v2`
- Store everything in the Chroma vector database under `chroma/`
- Build and persist a BM25 keyword index to `bm25_index.pkl`

> **Note:** Duplicate chunks are automatically detected and skipped, so you can safely re-run `populate_db.py` when adding new documents.

---

## Usage

### Querying the RAG System

```bash
python query_data.py "What is the Transformer architecture?"
```

The system will:
1. Retrieve the top 10 results each from vector search and BM25 (20 total)
2. Rerank them with the cross-encoder to select the top 5 most relevant chunks
3. Pass the chunks as context to `llama3.2` with a strict, citation-enforcing prompt
4. Validate that all citations in the response correspond to retrieved documents
5. Print the final, validated answer

**Example output:**
```
Response: The Transformer is a sequence transduction model that relies entirely on
self-attention to compute representations of its input and output [data:Attention_all_you_need.pdf:3:0].
It dispenses with recurrence and convolutions [data:Attention_all_you_need.pdf:3:1].
```

If a question cannot be answered from the available documents, the system responds:
```
Response: I cannot answer this based on provided documents.
```

### Evaluating Quality

Run the full RAGAS evaluation against the curated test dataset (`test.csv`):

```bash
python evaluate_rag.py
```

This computes RAGAS metrics (faithfulness, answer relevancy, context precision, context recall) for each question in the test set and saves the results to `final_evaluation_metrics.csv`. The script exits with code `0` if all quality gate thresholds are met, or `1` if any threshold is not met.

Alternatively, run the LLM-as-a-judge evaluation:

```bash
python evaluate_rag_llama.py
```

### Generating Test Data

Automatically generate Q&A pairs from your PDF documents:

```bash
python generate_candidates.py
```

This uses `llama3.2` to generate up to 100 question-answer pairs from the indexed document chunks and saves them to `candidate_dataset.csv`.

### Running Unit Tests

```bash
pytest tests/test_rag.py
```

---

## Evaluation & Quality Gates

The system is evaluated using [RAGAS](https://docs.ragas.io/en/stable/) metrics. The following thresholds must be met for the quality gate to pass:

| Metric | Threshold | Description |
|---|---|---|
| **Context Precision** | ≥ 0.70 | Are the retrieved chunks actually relevant to the question? |
| **Context Recall** | ≥ 0.40 | Are all necessary source passages being retrieved? |
| **Faithfulness** | ≥ 0.40 | Does the answer stay faithful to the source documents (no hallucinations)? |
| **Answer Relevancy** | ≥ 0.30 | Is the generated answer relevant to the question asked? |

Evaluation results are persisted to `final_evaluation_metrics.csv` for analysis and tracking.

---

## CI/CD Integration

The repository includes a GitHub Actions workflow (`.github/workflows/rag_eval.yml`) that runs automatically on every pull request to `main`. The workflow:

1. Sets up a Python 3.12 environment
2. Installs all dependencies
3. Runs `evaluate_rag.py` against the existing Chroma database
4. **Blocks the PR merge** if any quality gate threshold is not met

This ensures that no change that degrades retrieval or answer quality can be merged without review.

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/` | Directory containing source PDF documents |
| `CHROMA_PATH` | `chroma/` | Directory for the Chroma vector database |

Prompts used for generation and evaluation are stored in `prompts.json` and can be updated without changing the Python code.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

© 2026 Viraj Patel