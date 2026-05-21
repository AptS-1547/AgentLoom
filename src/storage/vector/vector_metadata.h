#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace agent::vector_storage {

/// Collection descriptor: logical grouping with shared fingerprint policy
struct CollectionDescriptor {
    std::string name;
    std::string embedding_model_fingerprint;
    std::string tokenizer_fingerprint;
    std::string pooling_strategy;  // "cls" | "mean"
    std::string normalization;     // "l2" | "none"
    std::size_t dimension = 0;
    std::string corpus_version;
    std::string policy_version;

    bool operator==(const CollectionDescriptor&) const = default;
};

/// Partition key: fine-grained physical isolation unit
struct PartitionKey {
    std::int64_t collection_id = 0;
    std::string tenant_id;       // empty = single-tenant
    std::string user_id;         // empty = global scope
    std::string memory_level;    // "L1"/"L2"/"L3"/"L4" or "working"/"state"/"episodic"/"knowledge"
    std::string scope_extras_json = "{}";  // JSON string for future extensions

    bool operator==(const PartitionKey&) const = default;
};

/// Hash functor for PartitionKey (for unordered_map)
struct PartitionKeyHash {
    std::size_t operator()(const PartitionKey& key) const noexcept;
};

/// Entry record: vector blob + metadata + lifecycle
struct EntryRecord {
    std::int64_t entry_id = 0;  // 0 = not assigned (for Insert)
    std::int64_t partition_id = 0;

    // Identity fields
    std::string cache_key;
    std::string text_hash;
    std::string content_hash;
    std::string memory_hash;

    // Vector (length must equal collection.dimension)
    std::vector<float> vector;

    // Lifecycle (aligned with Python MemoryLifecycleStore)
    std::int64_t recall_count = 0;
    std::optional<std::int64_t> last_recalled_at_ms;
    bool forgotten = false;
    std::optional<std::int64_t> forgotten_at_ms;
    std::optional<std::int64_t> deleted_after_ms;
    std::int64_t forget_epoch = 0;

    // Semantic metadata
    std::string memory_type;  // "state_snapshot"|"fact"|"profile"|"capability"|"visual"|...
    std::string emotion;
    float emotion_intensity = 0.0f;
    float state_arousal = 0.0f;
    std::string answer_type;  // reserved for Phase 5 semantic cache

    // Payload (actual answer / state snapshot JSON)
    std::string payload;

    // Extension slot (avoid frequent schema migrations)
    std::string extra_metadata_json = "{}";

    // Timestamps
    std::int64_t created_at_ms = 0;
    std::optional<std::int64_t> expires_at_ms;
};

/// Partition snapshot: metadata for hydration staleness detection
struct PartitionSnapshot {
    std::int64_t partition_id = 0;
    std::int64_t vector_count = 0;
    std::int64_t last_modified_at_ms = 0;
};

}  // namespace agent::vector_storage
