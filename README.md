# PDF Search (Hybrid BM25 + Vectors)

A backend service to **ingest parsed PDF data** (paragraphs, tables, images) and provide **search** via a hybrid of:

- 🔍 **Full-text search** (Postgres `tsvector` / BM25-like ranking)  
- 🤖 **Semantic search** (SentenceTransformers embeddings with pgvector)  
- ⚡ **Fusion** of lexical + vector scores for better relevance

---

## 🚀 Features

- Extracts **paragraphs, tables, and images** from PDFs (via your parser).
- ETL pipeline to load JSON into Postgres with embeddings.
- Search API built on **FastAPI** with:
  - API Key authentication
  - Hybrid (lexical + vector) scoring
  - Optional filters (`pdf_id`, `alpha` weight, etc.)
- Runs locally with **Docker Compose** (Postgres + pgvector).
- Ready for deployment on **AWS ECS + RDS + S3**.

---

## 📂 Project Structure

```
pdf-search/
├── api.py              # FastAPI search API
├── etl.py              # ETL pipeline (ingests parsed JSON)
├── schema.sql          # Postgres schema (with pgvector + FTS)
├── requirements.txt    # Python dependencies
├── docker-compose.yml  # Local Postgres + pgvector
├── .env.example        # Template for environment variables
├── .gitignore
├── README.md
└── sample/
    ├── sample.pdf
    └── sample_parsed.json   # Parsed output from your PDF parser
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/pdf-search.git
cd pdf-search
```

### 2. Start Postgres (with pgvector)
```bash
docker compose up -d
```

This will start a local **Postgres 16** instance with the `vector` extension enabled.

### 3. Virtual environment + dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Environment variables
Copy the example and fill in your secrets:
```bash
cp .env.example .env
```

Edit `.env` as needed (e.g., API_KEY, DB URL).

---

## 📥 ETL: Load parsed PDF data

Your parser should produce JSON shaped like:

```json
{
  "document_id": "doc-001",
  "title": "Sample",
  "num_pages": 1,
  "paragraphs": [
    {"page": 1, "bbox": [39.6, 376.5, 422.2, 465.7], "text": "Some text..."}
  ],
  "tables": [
    {"page": 1, "bbox": [45.6, 281.3, 416.2, 299.5], "cells": [["Header1","Header2"],["val1","val2"]]}
  ],
  "images": [
    {"page": 1, "bbox": null, "caption": "Figure 1...", "ocr_text": "Detected text"}
  ]
}
```

Ingest it into Postgres with embeddings:

```bash
python etl.py --json sample/sample_parsed.json --document-id doc-001 --title "Sample Document"
```

---

## 🔎 API: Run the search service

Start the FastAPI app:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Health check:
```bash
curl http://localhost:8000/healthz
# {"ok": true}
```

Search:
```bash
curl -G "http://localhost:8000/search" \
  --data-urlencode "q=heat treatment cracking" \
  --data-urlencode "pdf_id=doc-001" \
  -H "X-API-Key: <your_api_key>"
```

Response example:
```json
{
  "query": "heat treatment cracking",
  "pdf_id": "doc-001",
  "hits": [
    {
      "chunk_id": 42,
      "document_id": "doc-001",
      "page_number": 1,
      "type": "paragraph",
      "text": "The plot of inhibitor concentration ...",
      "bbox": {"x1": 39.68, "y1": 376.51, "x2": 422.23, "y2": 465.79},
      "score": 0.47
    }
  ]
}
```

---

## 🛡️ Security

- All requests require `X-API-Key` header.
- Never commit your real `.env`; only commit `.env.example`.

---

## 🛠️ Development Notes

- Schema (`schema.sql`) automatically sets up `documents` and `chunks` with indexes.
- `etl.py` chunks long paragraphs, normalizes whitespace, and computes embeddings.
- Adjust `alpha` (0–1) in API queries to balance lexical vs vector relevance.
- You can enforce **keyword-only** search by setting `alpha=1`.

---

## ☁️ Deployment (AWS)

- **DB**: Amazon RDS (Postgres with `pgvector`).
- **App**: ECS Fargate (API container).
- **Storage**: S3 for PDFs + parsed JSON.
- **Secrets**: AWS Secrets Manager for `DATABASE_URL` + `API_KEY`.
- **Networking**: ALB + ACM TLS + WAF.
- **Monitoring**: CloudWatch logs & metrics.