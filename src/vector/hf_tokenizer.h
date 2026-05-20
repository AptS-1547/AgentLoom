#pragma once

#include "result.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <vector>

namespace vector {

struct EncodeOptions {
    std::size_t max_length = 512;
    bool truncation = true;
    bool padding = true;
    bool pad_to_longest_in_batch = true;
    bool add_special_tokens = true;
};

struct TokenizedBatch {
    std::size_t batch_size = 0;
    std::size_t sequence_length = 0;
    std::vector<std::int64_t> input_ids;
    std::vector<std::int64_t> attention_mask;
    std::vector<std::int64_t> token_type_ids;
};

class HfTokenizer {
public:
    HfTokenizer() noexcept;
    ~HfTokenizer() noexcept;

    HfTokenizer(const HfTokenizer&) = delete;
    HfTokenizer& operator=(const HfTokenizer&) = delete;

    HfTokenizer(HfTokenizer&& other) noexcept;
    HfTokenizer& operator=(HfTokenizer&& other) noexcept;

    static core::Result<HfTokenizer> LoadFromFile(const std::filesystem::path& path);

    core::Result<HfTokenizer> Clone() const;

    bool valid() const noexcept;
    explicit operator bool() const noexcept { return valid(); }

    core::Result<TokenizedBatch> Encode(
        std::string_view text,
        const EncodeOptions& options = {}) const;

    core::Result<TokenizedBatch> EncodeBatch(
        std::span<const std::string_view> texts,
        const EncodeOptions& options = {}) const;

    static int AbiVersion() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vector
