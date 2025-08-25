import argparse, json, os, re, math
from typing import List, Dict
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------- Utilities ----------------
def normalize_ws(s: str) -> str:
    # collapse whitespace, keep paragraphs
    s = re.sub(r'\r\n?', '\n', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def flatten_table(cells: List[List[str]]) -> str:
    # join headers + rows into a linearized string for search
    rows = []
    for row in cells:
        rows.append(" | ".join([normalize_ws(str(c)) for c in row]))
    return "\n".join(rows)

def chunk_long_text(text: str, max_chars: int = 1800) -> List[str]:
    # safe chunking on sentence boundaries
    if len(text) <= max_chars:
        return [text]
    parts, buf = [], []
    size = 0
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sent in sentences:
        if size + len(sent) + 1 > max_chars and buf:
            parts.append(" ".join(buf).strip())
            buf, size = [], 0
        buf.append(sent)
        size += len(sent) + 1
    if buf:
        parts.append(" ".join(buf).strip())
    return parts

# ---------------- DB ----------------
pool = ConnectionPool(DB_URL, min_size=1, max_size=5, kwargs={"autocommit": True})

def upsert_document(doc_id: str, title: str, num_pages: int, s3_key: str | None = None):
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents(id, title, num_pages, s3_key)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET title=EXCLUDED.title, num_pages=EXCLUDED.num_pages, s3_key=EXCLUDED.s3_key
            """,
            (doc_id, title, num_pages, s3_key),
        )

def insert_chunk(doc_id: str, type_: str, page: int, bbox: Dict, text: str, emb: np.ndarray):
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks(document_id, type, page_number, bbox, text, tokens, embedding)
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
            """,
            (doc_id, type_, page, json.dumps(bbox), text, len(text.split()), emb.tolist()),
        )

# ---------------- ETL ----------------
def run_etl(parsed_json_path: str, document_id: str | None = None, title: str | None = None, s3_key: str | None = None):
    with open(parsed_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_id = document_id or data.get("document_id")
    if not doc_id:
        raise ValueError("document_id is required (either CLI param or in JSON).")
    title = title or data.get("title", doc_id)
    num_pages = int(data.get("num_pages", 0))

    print(f"[ETL] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[ETL] Upserting document {doc_id}")
    upsert_document(doc_id, title, num_pages, s3_key)

    def add_text_piece(type_, page, bbox, raw_text):
        raw_text = normalize_ws(raw_text)
        if not raw_text:
            return
        for part in chunk_long_text(raw_text):
            emb = model.encode([part], normalize_embeddings=True)[0]
            insert_chunk(doc_id, type_, page, bbox or {}, part, emb)

    # Paragraphs
    for p in data.get("paragraphs", []):
        add_text_piece("paragraph", int(p.get("page", 0)), p.get("bbox", {}), p.get("text",""))

    # Tables
    for t in data.get("tables", []):
        flat = flatten_table(t.get("cells", []))
        # You can also store the full table JSON separately if desired
        add_text_piece("table", int(t.get("page", 0)), t.get("bbox", {}), flat)

    # Images (caption + OCR)
    for im in data.get("images", []):
        if im.get("caption"):
            add_text_piece("caption", int(im.get("page", 0)), im.get("bbox", {}), im.get("caption"))
        if im.get("ocr_text"):
            add_text_piece("image_ocr", int(im.get("page", 0)), im.get("bbox", {}), im.get("ocr_text"))

    print("[ETL] Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to parsed JSON from your PDF parser")
    ap.add_argument("--document-id", required=False)
    ap.add_argument("--title", required=False)
    ap.add_argument("--s3-key", required=False, help="S3 key of the original PDF (optional)")
    args = ap.parse_args()
    run_etl(args.json, args.document_id, args.title, args.s3_key)
