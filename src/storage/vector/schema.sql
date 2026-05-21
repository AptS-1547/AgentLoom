-- Phase 4 Vector Storage Schema
-- SQLite 唯一真源，Faiss 作为懒加载 hydrated cache
-- 设计原则：纯标准 SQL，细粒度分区键，tenant_id 从 day 1 进 schema

-- 1. 集合：共享 fingerprint 策略的逻辑组
CREATE TABLE IF NOT EXISTS vector_collections (
    collection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,                   -- e.g. "memory_v1", "rag_card_v1"
    embedding_model_fingerprint TEXT NOT NULL,   -- sha256 of (model_path + dim + pooling + normalize)
    tokenizer_fingerprint TEXT NOT NULL,         -- sha256 of tokenizer.json
    pooling_strategy TEXT NOT NULL,              -- "cls" | "mean"
    normalization TEXT NOT NULL,                 -- "l2" | "none"
    dimension INTEGER NOT NULL,
    corpus_version TEXT NOT NULL,                -- 业务侧维护，区分语料版本
    policy_version TEXT NOT NULL,                -- 业务侧维护，区分缓存策略版本
    created_at_ms INTEGER NOT NULL
);

-- 2. 分区：物理隔离单元，细粒度复合键
CREATE TABLE IF NOT EXISTS vector_partitions (
    partition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES vector_collections(collection_id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL DEFAULT '',          -- 单租户场景空串
    user_id TEXT NOT NULL DEFAULT '',            -- '' 表示 global 范围
    memory_level TEXT NOT NULL,                  -- "L1"|"L2"|"L3"|"L4" 或 "working"|"state"|"episodic"|"knowledge"
    scope_extras TEXT NOT NULL DEFAULT '{}',     -- JSON 字符串：persona_id / course_id 等未来扩展
    vector_count INTEGER NOT NULL DEFAULT 0,     -- 维护计数器，避免每次 COUNT(*)
    last_modified_at_ms INTEGER NOT NULL,        -- 用于 Index Manager 判断 hydrated 是否过期
    created_at_ms INTEGER NOT NULL,
    UNIQUE(collection_id, tenant_id, user_id, memory_level)
);

CREATE INDEX IF NOT EXISTS idx_partitions_collection ON vector_partitions(collection_id);
CREATE INDEX IF NOT EXISTS idx_partitions_lookup ON vector_partitions(collection_id, tenant_id, user_id, memory_level);

-- 3. 条目：向量 blob + 元数据 + 生命周期，唯一真源
CREATE TABLE IF NOT EXISTS vector_entries (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 同时用作 Faiss IDMap2 id
    partition_id INTEGER NOT NULL REFERENCES vector_partitions(partition_id) ON DELETE CASCADE,

    -- 标识字段
    cache_key TEXT NOT NULL,                     -- 应用层去重 key
    text_hash TEXT NOT NULL,                     -- sha256 of source text
    content_hash TEXT NOT NULL,                  -- sha256 of (user_id + content)
    memory_hash TEXT NOT NULL,                   -- 旧 Python 系统的 stable_memory_hash

    -- 向量
    vector BLOB NOT NULL,                        -- float32[dim]，字节长度 = dim * 4

    -- 生命周期（对齐 Python MemoryLifecycleStore）
    recall_count INTEGER NOT NULL DEFAULT 0,
    last_recalled_at_ms INTEGER,                 -- NULL = 从未召回
    forgotten INTEGER NOT NULL DEFAULT 0,        -- 0/1
    forgotten_at_ms INTEGER,
    deleted_after_ms INTEGER,                    -- 物理删除时间点
    forget_epoch INTEGER NOT NULL DEFAULT 0,

    -- 语义元数据
    memory_type TEXT NOT NULL DEFAULT '',        -- "state_snapshot"|"fact"|"profile"|"capability"|"visual"|...
    emotion TEXT NOT NULL DEFAULT '',
    emotion_intensity REAL NOT NULL DEFAULT 0.0,
    state_arousal REAL NOT NULL DEFAULT 0.0,
    answer_type TEXT NOT NULL DEFAULT '',        -- 留给 Phase 5 语义缓存

    -- payload（实际的回答 / 状态快照 JSON）
    payload TEXT NOT NULL DEFAULT '',

    -- 扩展位（避免后续频繁 schema 迁移）
    extra_metadata TEXT NOT NULL DEFAULT '{}',   -- JSON 字符串

    -- 时间戳
    created_at_ms INTEGER NOT NULL,
    expires_at_ms INTEGER,                       -- TTL，NULL = 永不过期

    UNIQUE(partition_id, memory_hash)
);

CREATE INDEX IF NOT EXISTS idx_entries_partition ON vector_entries(partition_id);
CREATE INDEX IF NOT EXISTS idx_entries_active ON vector_entries(partition_id, forgotten);
CREATE INDEX IF NOT EXISTS idx_entries_content ON vector_entries(partition_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_entries_expiry ON vector_entries(expires_at_ms) WHERE expires_at_ms IS NOT NULL;
