#pragma once

#include <cstddef>
#include <string>

namespace agent::vector_storage {

/// Fingerprint computation helpers.
///
/// Fingerprints lock down "what an embedding means" — the model file, dimension,
/// pooling strategy, normalization, and tokenizer.  If any of these change, the
/// existing on-disk vectors are no longer queryable with new query vectors,
/// because they live in a different semantic space.
///
/// The collection schema stores fingerprints so that we can detect this mismatch
/// at startup and refuse to silently corrupt search results.  Callers compare
/// two `CollectionDescriptor` values directly via `operator==`.
namespace VectorFingerprint {

/// Compute an embedding model fingerprint from its loaded properties.
/// Uses model_path (the source artifact), dimension, pooling strategy and
/// normalization mode to produce a stable SHA-256 hex digest.
std::string ComputeEmbeddingFingerprint(const std::string& model_path,
                                        std::size_t dimension,
                                        const std::string& pooling_strategy,
                                        const std::string& normalization);

/// Compute a tokenizer fingerprint by hashing the tokenizer.json file contents.
/// Returns empty string if the file cannot be read.
std::string ComputeTokenizerFingerprint(const std::string& tokenizer_json_path);

}  // namespace VectorFingerprint

}  // namespace agent::vector_storage
