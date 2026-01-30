-- ComfyUI Memory RAG v1.1 Schema
-- Optimized for pgvector & Data Integrity

-- 1. 필수 확장 설치
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. workflows 테이블
CREATE TABLE IF NOT EXISTS workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- SHA-256 전체(64자) 저장으로 충돌 방지
    workflow_hash VARCHAR(64) NOT NULL UNIQUE,
    
    workflow_json JSONB NOT NULL,
    full_text TEXT NOT NULL DEFAULT '',
    
    -- 검색 필터용 모델 목록 (GIN 인덱스용)
    models TEXT[] DEFAULT '{}',
    
    -- 메타데이터
    name VARCHAR(255),
    tags TEXT[] DEFAULT '{}',
    success_count INTEGER DEFAULT 0,
    
    -- 임베딩 (all-MiniLM-L6-v2: 384 dim)
    embedding vector(384),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_workflows_hash ON workflows(workflow_hash);
CREATE INDEX IF NOT EXISTS idx_workflows_created ON workflows(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_workflows_models ON workflows USING GIN(models); -- Array 검색 최적화
CREATE INDEX IF NOT EXISTS idx_workflows_tags ON workflows USING GIN(tags);

-- HNSW 인덱스 (Cosine Distance 최적화)
-- vector_cosine_ops 명시 필수 (<=> 연산자용)
CREATE INDEX IF NOT EXISTS idx_workflows_embedding ON workflows 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);


-- 3. generations 테이블
CREATE TABLE IF NOT EXISTS generations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    
    -- 클라이언트 실행 ID (재시도 중복 방지)
    client_run_id VARCHAR(64) NOT NULL UNIQUE,
    
    prompt_positive TEXT,
    prompt_negative TEXT,
    
    -- 파라미터
    seed BIGINT,
    cfg_scale REAL,
    steps INTEGER,
    sampler VARCHAR(64),
    scheduler VARCHAR(64),
    
    -- 결과물
    width INTEGER,
    height INTEGER,
    fps INTEGER,
    duration_sec REAL,
    result_paths TEXT[] DEFAULT '{}',
    thumbnail_path TEXT,
    
    status VARCHAR(32) DEFAULT 'success',
    error_log TEXT,
    render_time_sec REAL,
    
    -- 프롬프트 임베딩
    embedding vector(384),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_generations_workflow ON generations(workflow_id);
CREATE INDEX IF NOT EXISTS idx_generations_client_id ON generations(client_run_id);
CREATE INDEX IF NOT EXISTS idx_generations_created ON generations(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_generations_embedding ON generations 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);


-- 4. recall_outbox 테이블
CREATE TABLE IF NOT EXISTS recall_outbox (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID NOT NULL UNIQUE DEFAULT uuid_generate_v4(),
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    
    workflow_json JSONB NOT NULL,
    overrides JSONB DEFAULT '{}',
    target_host VARCHAR(255) DEFAULT 'http://192.168.1.100:8188',
    
    status VARCHAR(32) DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    last_attempt_at TIMESTAMP WITH TIME ZONE,
    error_log TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sent_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_outbox_status ON recall_outbox(status);


-- 5. Updated_at 자동 갱신 트리거
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_workflows_updated ON workflows;
CREATE TRIGGER trigger_workflows_updated
    BEFORE UPDATE ON workflows
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trigger_recall_outbox_updated ON recall_outbox;
CREATE TRIGGER trigger_recall_outbox_updated
    BEFORE UPDATE ON recall_outbox
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- 6. 성공 횟수 자동 증가 트리거
CREATE OR REPLACE FUNCTION increment_workflow_success()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'success' THEN
        UPDATE workflows
        SET success_count = success_count + 1
        WHERE id = NEW.workflow_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_increment_success ON generations;
CREATE TRIGGER trigger_increment_success
    AFTER INSERT ON generations
    FOR EACH ROW
    EXECUTE FUNCTION increment_workflow_success();