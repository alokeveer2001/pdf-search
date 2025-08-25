import os, hashlib, hmac
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Query, Header, status
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg
from psycopg_pool import ConnectionPool
from fastapi.middleware.cors import CORSMiddleware
from typing import Union, Dict, List, Optional

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --------- Security helpers ----------
def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    if not API_KEY or not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key")
    # constant-time compare
    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key")
    return True

# --------- App & middlewares ----------
app = FastAPI(title="PDF Search API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # set allowed origins in prod, e.g. ["https://your-frontend.com"]
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# --------- DB & model ----------
pool = ConnectionPool(DB_URL, min_size=1, max_size=10, kwargs={"autocommit": True})
model = SentenceTransformer(MODEL_NAME)

# --------- Schemas ----------
class SearchHit(BaseModel):
    chunk_id: int
    document_id: str
    page_number: int
    type: str
    text: str
    bbox: Optional[Union[Dict[str, float], List[float]]] = None
    score: float

class SearchResponse(BaseModel):
    query: str
    pdf_id: Optional[str] = None
    hits: List[SearchHit]

# --------- Search ----------
def hybrid_search(q: str, pdf_id: Optional[str], k: int = 20, alpha: float = 0.5):
    q_text = q.strip()
    if not q_text:
        return []

    # Compute embedding once
    q_emb = model.encode([q_text], normalize_embeddings=True)[0].tolist()

    # Build params in correct order depending on pdf_id
    if pdf_id:
        params = [
            q_text,      # 1) ts_rank_cd(... %s)
            pdf_id,      # 2) WHERE c.document_id = %s AND ...
            q_text,      # 3) ... @@ plainto_tsquery('english', %s)
            q_emb,       # 4) SELECT 1 - (embedding <=> %s::vector)
            pdf_id,      # 5) WHERE c.document_id = %s AND TRUE
            q_emb,       # 6) ORDER BY embedding <-> %s::vector
            alpha,       # 7) fused weight (lex)
            1.0 - alpha, # 8) fused weight (vec)
            k,           # 9) LIMIT
        ]
        doc_filter = "c.document_id = %s AND "
    else:
        params = [
            q_text,      # 1) ts_rank_cd(... %s)
            q_text,      # 2) ... @@ plainto_tsquery('english', %s)
            q_emb,       # 3) SELECT 1 - (embedding <=> %s::vector)
            q_emb,       # 4) ORDER BY embedding <-> %s::vector
            alpha,       # 5) fused weight (lex)
            1.0 - alpha, # 6) fused weight (vec)
            k,           # 7) LIMIT
        ]
        doc_filter = ""

    sql = f"""
    WITH
      lex AS (
        SELECT
          c.id as chunk_id, c.document_id, c.page_number, c.type, c.text, c.bbox,
          ts_rank_cd(c.tsv, plainto_tsquery('english', %s)) AS score
        FROM chunks c
        WHERE {doc_filter} c.tsv @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT 50
      ),
      vec AS (
        SELECT
          c.id as chunk_id, c.document_id, c.page_number, c.type, c.text, c.bbox,
          1 - (c.embedding <=> %s::vector) AS score
        FROM chunks c
        WHERE {doc_filter} TRUE
        ORDER BY c.embedding <-> %s::vector
        LIMIT 50
      ),
      allhits AS (
        SELECT *,'lex' AS kind FROM lex
        UNION ALL
        SELECT *,'vec' AS kind FROM vec
      ),
      fused AS (
        SELECT
          chunk_id, document_id, page_number, type, text, bbox,
          MAX(CASE WHEN kind='lex' THEN score ELSE 0 END) * %s
          + MAX(CASE WHEN kind='vec' THEN score ELSE 0 END) * %s
          AS fused_score
        FROM allhits
        GROUP BY chunk_id, document_id, page_number, type, text, bbox
      )
    SELECT chunk_id, document_id, page_number, type, text, bbox, fused_score
    FROM fused
    ORDER BY fused_score DESC
    LIMIT %s;
    """

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        return [
            dict(
                chunk_id=r[0],
                document_id=r[1],
                page_number=r[2],
                type=r[3],
                text=r[4],
                bbox=r[5],
                score=float(r[6]),
            )
            for r in rows
        ]

# --------- Endpoints ----------
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, max_length=500),
    pdf_id: Optional[str] = Query(default=None),
    k: int = Query(default=20, ge=1, le=100),
    alpha: float = Query(default=0.55, ge=0.0, le=1.0),
    _: bool = Depends(verify_api_key),
):
    hits = hybrid_search(q, pdf_id, k=k, alpha=alpha)
    return {"query": q, "pdf_id": pdf_id, "hits": hits}

# --------- Graceful shutdown ----------
@app.on_event("shutdown")
def _shutdown():
    try:
        pool.close()
    except Exception:
        pass
