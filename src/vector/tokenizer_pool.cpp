#include "tokenizer_pool.h"

#include <utility>

namespace vector {

TokenizerLease::TokenizerLease(
    std::unique_ptr<HfTokenizer> tokenizer,
    std::weak_ptr<core::BlockingQueue<std::unique_ptr<HfTokenizer>>> owner) noexcept
    : tokenizer_(std::move(tokenizer)), owner_(std::move(owner)) {}

TokenizerLease::TokenizerLease(TokenizerLease&& other) noexcept
    : tokenizer_(std::move(other.tokenizer_)), owner_(std::move(other.owner_)) {}

TokenizerLease& TokenizerLease::operator=(TokenizerLease&& other) noexcept {
    if (this != &other) {
        Release();
        tokenizer_ = std::move(other.tokenizer_);
        owner_ = std::move(other.owner_);
    }
    return *this;
}

void TokenizerLease::Release() noexcept {
    if (!tokenizer_) {
        return;
    }
    if (auto queue = owner_.lock()) {
        // If the pool was closed between acquire and release, TryPush will
        // return Cancelled. We then drop the tokenizer locally; the pool
        // capacity is permanently reduced, which is fine on shutdown.
        auto status = queue->TryPush(std::move(tokenizer_));
        (void)status;
    }
    tokenizer_.reset();
    owner_.reset();
}

TokenizerPool::TokenizerPool(std::size_t size,
                             std::chrono::milliseconds default_timeout) noexcept
    : size_(size), default_timeout_(default_timeout) {}

TokenizerPool::~TokenizerPool() noexcept {
    Close();
}

core::Result<std::shared_ptr<TokenizerPool>> TokenizerPool::Create(
    const HfTokenizer& source,
    TokenizerPoolOptions options) {
    if (options.size == 0) {
        return core::Status(core::ErrorCode::InvalidArgument,
                            "TokenizerPool::Create: size must be >= 1");
    }
    if (!source.valid()) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "TokenizerPool::Create: source tokenizer is invalid");
    }

    std::shared_ptr<TokenizerPool> pool(
        new TokenizerPool(options.size, options.default_acquire_timeout));
    pool->queue_ = std::make_shared<Queue>(options.size);

    for (std::size_t i = 0; i < options.size; ++i) {
        auto clone_result = source.Clone();
        if (!clone_result.ok()) {
            return core::Status(clone_result.status().code(),
                                std::string("TokenizerPool::Create: clone failed: ")
                                    + clone_result.status().message());
        }
        auto tk = std::make_unique<HfTokenizer>(std::move(clone_result).value());
        auto push_status = pool->queue_->TryPush(std::move(tk));
        if (!push_status.ok()) {
            return core::Status(push_status.code(),
                                std::string("TokenizerPool::Create: TryPush failed: ")
                                    + push_status.message());
        }
    }
    return pool;
}

core::Result<TokenizerLease> TokenizerPool::Acquire() {
    return Acquire(default_timeout_);
}

core::Result<TokenizerLease> TokenizerPool::Acquire(std::chrono::milliseconds timeout) {
    if (!queue_) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "TokenizerPool::Acquire: pool is not initialized");
    }
    if (queue_->closed()) {
        return core::Status(core::ErrorCode::Cancelled,
                            "TokenizerPool::Acquire: pool is closed");
    }
    auto popped = queue_->WaitPopFor(timeout);
    if (!popped.ok()) {
        return popped.status();
    }
    return TokenizerLease(std::move(popped).value(), queue_);
}

core::Result<TokenizerLease> TokenizerPool::TryAcquire() {
    if (!queue_) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "TokenizerPool::TryAcquire: pool is not initialized");
    }
    if (queue_->closed()) {
        return core::Status(core::ErrorCode::Cancelled,
                            "TokenizerPool::TryAcquire: pool is closed");
    }
    auto popped = queue_->TryPop();
    if (!popped.ok()) {
        return popped.status();
    }
    return TokenizerLease(std::move(popped).value(), queue_);
}

void TokenizerPool::Close() noexcept {
    if (queue_) {
        queue_->Close(/*discard=*/true);
    }
}

bool TokenizerPool::closed() const noexcept {
    return queue_ ? queue_->closed() : true;
}

std::size_t TokenizerPool::available() const noexcept {
    return queue_ ? queue_->size() : 0;
}

} // namespace vector
