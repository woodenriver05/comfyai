-- ComfyUI Memory RAG v1.1 - Documents Extension
-- Agent Knowledge Base (Worklogs, Architecture, Snippets)

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Content Hash for Deduplication
    -- sha256(source_path + chunk_index + content)
    doc_hash VARCHAR(64) NOT NULL UNIQUE,
    
    -- Metadata
    type VARCHAR(32) NOT NULL,  -- worklog, architecture, decision, snippet
    title TEXT NOT NULL,
    source_path TEXT,           -- file path (e.g., docs/WORKLOG.md)
    chunk_index INTEGER DEFAULT 0,
    
    -- Content
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    
    -- Embedding (384 dim)
    embedding vector(384),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(doc_hash);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_tags ON documents USING GIN(tags);

-- HNSW Index for Semantic Search
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Trigger for updated_at
DROP TRIGGER IF EXISTS trigger_documents_updated ON documents;
CREATE TRIGGER trigger_documents_updated
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();