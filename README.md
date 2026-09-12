# Production GenAI — RAG & Agent Platform

A production-oriented Retrieval-Augmented Generation (RAG) platform built with **Python, FastAPI, Qdrant, BM25, Cross-Encoder reranking, LangGraph, OpenRouter, Docker, and pytest**.

The project is designed to demonstrate how a RAG system can move beyond a basic "PDF chatbot" into a modular, testable, evaluated, API-driven application.

---

## Overview

This system answers questions from indexed documents using a multi-stage retrieval pipeline:

```text
Documents
   │
   ▼
Ingestion & Chunking
   │
   ▼
Embeddings ───────────────► Qdrant
   │                         │
   │                         ▼
   └────────────────────► Dense Retrieval
                              │
                              ├──────────────┐
                              │              │
                              ▼              ▼
                         BM25 Search    Dense Search
                              │              │
                              └──────┬───────┘
                                     ▼
                              RRF Hybrid Fusion
                                     │
                                     ▼
                            Cross-Encoder Reranker
                                     │
                                     ▼
                              Context Builder
                                     │
                                     ▼
                               LangGraph Agent
                                     │
                                     ▼
                              OpenRouter LLM
                                     │
                                     ▼
                         Grounded Answer + Sources
                                     │
                                     ▼
                                  FastAPI
```

---

## Key Features

- **PDF ingestion** with page-level metadata
- **Recursive chunking** with configurable chunk size and overlap
- **Sentence-transformer embeddings** using `all-MiniLM-L6-v2`
- **Persistent Qdrant vector database**
- **BM25 sparse retrieval**
- **Hybrid retrieval** using Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking**
- **LangGraph orchestration**
- **Grounded generation** with strict document-only prompting
- **Citation-aware responses**
- **Relevance gate / abstention** for unsupported questions
- **FastAPI REST API**
- **Swagger/OpenAPI documentation**
- **Structured application logging**
- **Health endpoint**
- **Docker and Docker Compose**
- **Automated pytest test suite**
- **Retrieval evaluation with Recall@K and MRR**
- **No LLM calls required for the retrieval benchmark**

---

## Retrieval Architecture

### 1. Dense Retrieval

Documents are embedded using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The resulting 384-dimensional vectors are stored in Qdrant using cosine similarity.

Dense retrieval provides semantic matching, allowing the system to retrieve relevant content even when the query and document use different wording.

### 2. BM25 Retrieval

BM25 provides lexical retrieval over the indexed document chunks.

This is useful for:

- exact terminology
- technical phrases
- numbers
- names
- keywords

### 3. Hybrid Retrieval

Dense and BM25 results are combined using **Reciprocal Rank Fusion (RRF)**.

The goal is to combine semantic and lexical retrieval instead of relying on a single retrieval strategy.

### 4. Cross-Encoder Reranking

The hybrid candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The cross-encoder scores the relationship between:

```text
query ↔ document chunk
```

This provides a more precise final ranking before context is sent to the LLM.

---

## Agent / RAG Flow

The LangGraph pipeline currently follows:

```text
START
  │
  ▼
Retrieve
  │
  ▼
Rerank
  │
  ▼
Generate
  │
  ▼
END
```

The application initializes the retrieval and generation components once at startup rather than rebuilding the entire pipeline for every request.

---

## Grounded Generation

The generation layer is explicitly instructed to use only the retrieved document context.

The system is designed to:

- avoid unsupported claims
- avoid answering unrelated questions using outside knowledge
- avoid inventing citations
- return an abstention response when sufficient evidence is unavailable

For unsupported questions, the system returns:

```text
I don't have enough information in the provided documents to answer this question.
```

This provides a basic hallucination-control mechanism through retrieval relevance gating and constrained prompting.

---

## Evaluation

The retrieval pipeline was evaluated on an 8-question manually labeled benchmark.

### Results

| Method | Recall@1 | Recall@3 | Recall@5 | MRR | Avg. Latency |
|---|---:|---:|---:|---:|---:|
| Dense | 0.292 | 0.292 | 0.292 | 0.500 | 70.80 ms |
| BM25 | 0.229 | 0.479 | 0.833 | 0.552 | 0.18 ms |
| Hybrid | 0.292 | 0.521 | 0.646 | 0.647 | 16.30 ms |
| Reranker | **0.521** | **0.667** | **0.917** | **0.938** | 427.95 ms |

### Interpretation

The benchmark shows the benefit of adding a learned reranking stage after candidate retrieval.

The final reranked pipeline achieved:

- **91.7% Recall@5**
- **0.938 MRR**

The reranker substantially improved the ordering of relevant evidence compared with the retrieval-only approaches.

The benchmark script does not make LLM calls, keeping retrieval evaluation deterministic and inexpensive.

---

## API

The application exposes a FastAPI service.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Query

```http
POST /query
```

Example request:

```json
{
  "question": "What is the highest-ranked page at a damping factor of 0.85?"
}
```

Example response:

```json
{
  "question": "What is the highest-ranked page at a damping factor of 0.85?",
  "answer": "Page C is the highest-ranked page at a damping factor of 0.85, with PageRank ≈ 0.2399. [Source 1]",
  "sources": [
    {
      "source": "fds_assessment__7__26MAG0065.pdf",
      "page": 12,
      "rerank_score": 6.81
    }
  ],
  "attempts": 0
}
```

The exact reranking scores may vary slightly depending on the runtime/model environment.

---

## Swagger UI

When running locally:

```text
http://localhost:8000/docs
```

The Swagger interface provides interactive documentation and allows API requests to be tested directly from the browser.

---

## Project Structure

```text
production-genai/
│
├── app/
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── process.py
│   │   ├── storage.py
│   │   └── index.py
│   │
│   ├── chunking/
│   │   └── recursive.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   │
│   ├── llm/
│   │   ├── client.py
│   │   ├── context.py
│   │   └── prompts.py
│   │
│   ├── agent/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── decision.py
│   │   └── graph.py
│   │
│   ├── evaluation/
│   │   ├── dataset.json
│   │   ├── metrics.py
│   │   ├── evaluate_retrieval.py
│   │   └── build_dataset.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routes/
│   │       └── query.py
│   │
│   ├── core/
│   │   └── logging.py
│   │
│   └── rag.py
│
├── tests/
│   ├── test_health.py
│   ├── test_metrics.py
│   └── test_validation.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/revathikrishapk/production-genai.git
cd production-genai
```

### 2. Create a virtual environment

Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```cmd
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Never commit `.env` or API keys to GitHub.

### 5. Build the document index

Place source documents in:

```text
data/raw/
```

Then run the indexing pipeline:

```cmd
python -m app.ingestion.index
```

This processes the documents, creates embeddings, and stores vectors in Qdrant.

---

## Run the API Locally

```cmd
uvicorn app.api.main:app --reload
```

Then open:

```text
http://localhost:8000/docs
```

---

## Docker

Build the image:

```cmd
docker compose build
```

Start the service:

```cmd
docker compose up -d
```

Check the service:

```cmd
docker compose ps
```

View logs:

```cmd
docker compose logs -f
```

Stop the service:

```cmd
docker compose down
```

The API will be available at:

```text
http://localhost:8000
```

---

## Testing

The project uses pytest.

Run:

```cmd
pytest -v
```

Current test suite:

```text
8 passed
```

Tests cover:

- health endpoint behavior
- Recall@K
- Reciprocal Rank
- empty-question validation
- placeholder-question validation
- valid-question validation

The pytest configuration restricts test discovery to the `tests/` directory so manual scripts do not accidentally execute during collection.

---

## Design Decisions

### Why Hybrid Retrieval?

Dense retrieval is strong at semantic similarity, while BM25 is strong at lexical matching.

Combining both improves robustness across different query types.

### Why Reranking?

Initial retrieval is optimized for finding a candidate set.

A cross-encoder can then perform a more expensive query-document relevance calculation on a smaller candidate set.

This creates a common production pattern:

```text
Fast candidate retrieval
        ↓
Smaller candidate set
        ↓
Expensive high-quality reranking
```

### Why LangGraph?

LangGraph provides an explicit graph-based orchestration layer.

This makes it easier to extend the system with additional states and decisions such as:

```text
Retrieve
   ↓
Rerank
   ↓
Check relevance
   ↓
Generate
   ↓
Retry / Rewrite
```

The current graph intentionally keeps the execution path simple while preserving an architecture that can be expanded.

### Why FastAPI?

FastAPI provides:

- typed request/response schemas
- automatic OpenAPI documentation
- validation
- easy containerization
- straightforward deployment

---

## Reliability & Production Considerations

The project includes several production-oriented practices:

- startup initialization of expensive models
- explicit Qdrant client lifecycle management
- structured logging
- health endpoint
- request validation
- relevance-based abstention
- source metadata in responses
- automated tests
- deterministic retrieval evaluation
- Dockerized deployment
- environment-based secret configuration

---

## Cost Considerations

The LLM layer is implemented through **OpenRouter** and configured to use a free model route by default.

The retrieval benchmark does not call the LLM, so evaluation can be run without consuming LLM quota.

Embedding and reranking are performed locally using Sentence Transformers.

---

## Limitations

This project is intentionally compact and has several areas for future production hardening:

- Qdrant currently uses a local persistent store for the local deployment
- the BM25 index is built in memory at application startup
- authentication and authorization are not implemented
- rate limiting is not implemented
- distributed tracing is not implemented
- document ingestion is currently batch-oriented
- the evaluation dataset is relatively small
- LLM output evaluation is not yet included
- production deployment requires external persistent vector storage

---

## Future Improvements

Potential next steps include:

- Qdrant Cloud / managed vector database
- background document ingestion
- asynchronous processing
- query rewriting and retrieval retry loops
- metadata filtering
- hybrid-score calibration
- larger evaluation datasets
- RAGAS / LLM-based evaluation
- OpenTelemetry tracing
- Prometheus metrics
- API authentication
- rate limiting
- CI/CD with GitHub Actions
- production cloud deployment
- frontend chat interface
- streaming responses

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| API | FastAPI |
| Vector DB | Qdrant |
| Dense Embeddings | Sentence Transformers |
| Sparse Retrieval | BM25 |
| Fusion | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| Agent Orchestration | LangGraph |
| LLM Gateway | OpenRouter |
| Containerization | Docker / Docker Compose |
| Testing | pytest |
| Document Processing | pypdf |

---

## Author

**Revathi Krishna**

GitHub:  
https://github.com/revathikrishapk

---

## License

This project is intended as a portfolio and engineering demonstration project.
