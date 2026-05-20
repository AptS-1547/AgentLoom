#pragma once

#include "blocking_queue.h"
#include "hf_tokenizer.h"
#include "result.h"

#include <chrono>
#include <cstddef>
#include <memory>

namespace vector {

class TokenizerPool;

struct TokenizerPoolOptions {
    std::size_t size = 4;
    std::chrono::milliseconds default_acquire_timeout{std::chrono::milliseconds{1000}};
};

class TokenizerLease {
public:
    TokenizerLease() noexcept = default;
    ~TokenizerLease() noexcept { Release(); }

    TokenizerLease(const TokenizerLease&) = delete;
    TokenizerLease& operator=(const TokenizerLease&) = delete;

    TokenizerLease(TokenizerLease&& other) noexcept;
    TokenizerLease& operator=(TokenizerLease&& other) noexcept;

    bool valid() const noexcept { return tokenizer_ != nullptr; }
    explicit operator bool() const noexcept { return valid(); }

    HfTokenizer* operator->() noexcept { return tokenizer_.get(); }
    const HfTokenizer* operator->() const noexcept { return tokenizer_.get(); }

    HfTokenizer& operator*() noexcept { return *tokenizer_; }
    const HfTokenizer& operator*() const noexcept { return *tokenizer_; }

    HfTokenizer* get() noexcept { return tokenizer_.get(); }
    const HfTokenizer* get() const noexcept { return tokenizer_.get(); }

    void Release() noexcept;

private:
    friend class TokenizerPool;

    TokenizerLease(std::unique_ptr<HfTokenizer> tokenizer,
                   std::weak_ptr<core::BlockingQueue<std::unique_ptr<HfTokenizer>>> owner) noexcept;

    std::unique_ptr<HfTokenizer> tokenizer_;
    std::weak_ptr<core::BlockingQueue<std::unique_ptr<HfTokenizer>>> owner_;
};

class TokenizerPool {
public:
    TokenizerPool() noexcept = default;
    ~TokenizerPool() noexcept;

    TokenizerPool(const TokenizerPool&) = delete;
    TokenizerPool& operator=(const TokenizerPool&) = delete;

    static core::Result<std::shared_ptr<TokenizerPool>> Create(
        const HfTokenizer& source,
        TokenizerPoolOptions options);

    core::Result<TokenizerLease> Acquire();
    core::Result<TokenizerLease> Acquire(std::chrono::milliseconds timeout);
    core::Result<TokenizerLease> TryAcquire();

    void Close() noexcept;
    bool closed() const noexcept;
    std::size_t size() const noexcept { return size_; }
    std::size_t available() const noexcept;

private:
    TokenizerPool(std::size_t size,
                  std::chrono::milliseconds default_timeout) noexcept;

    using Queue = core::BlockingQueue<std::unique_ptr<HfTokenizer>>;

    std::shared_ptr<Queue> queue_;
    std::size_t size_ = 0;
    std::chrono::milliseconds default_timeout_{1000};
};

} // namespace vector
