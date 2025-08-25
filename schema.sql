CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id           TEXT PRIMARY KEY,
  title        TEXT,
  s3_key       TEXT,
  num_pages    INT,
  created_at   TIMESTAMPTZ DEFAULT now()
);

-- 384 dims for all-MiniLM-L6-v2
CREATE TABLE IF NOT EXISTS chunks (
  id           BIGSERIAL PRIMARY KEY,
  document_id  TEXT REFERENCES documents(id) ON DELETE CASCADE,
  type         TEXT CHECK (type IN ('paragraph','table','image_ocr','caption')),
  page_number  INT,
  bbox         JSONB,
  text         TEXT NOT NULL,
  tokens       INT,
  embedding    VECTOR(384),
  tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(text,''))) STORED
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
-- Vector index (IVFFlat). You must ANALYZE on big datasets; adjust lists.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
