#pragma once

#include "result.h"

#include <cstddef>
#include <memory>

namespace core {

class RawMemoryPool;

struct MemoryPoolStats {
    std::size_t allocated_bytes = 0;
    std::size_t free_bytes = 0;
    std::size_t allocations = 0;
    std::size_t deallocations = 0;
    std::size_t peak_allocated_bytes = 0;
};

class MemoryBlock {
public:
    MemoryBlock() noexcept = default;
    MemoryBlock(MemoryBlock&& other) noexcept;
    MemoryBlock& operator=(MemoryBlock&& other) noexcept;
    ~MemoryBlock();

    MemoryBlock(const MemoryBlock&) = delete;
    MemoryBlock& operator=(const MemoryBlock&) = delete;

    std::byte* data() noexcept;
    const std::byte* data() const noexcept;
    std::size_t size() const noexcept;
    std::size_t alignment() const noexcept;
    RawMemoryPool* owner() const noexcept;
    bool empty() const noexcept;

    std::byte* release() noexcept;
    void reset() noexcept;

private:
    friend class RawMemoryPool;
    friend class BucketMemoryPool;

    MemoryBlock(RawMemoryPool* owner, std::byte* data, std::size_t size, std::size_t alignment) noexcept;

    RawMemoryPool* owner_ = nullptr;
    std::byte* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t alignment_ = 0;
};

class RawMemoryPool {
public:
    virtual ~RawMemoryPool() = default;

    virtual Result<MemoryBlock> allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) = 0;
    virtual void release(std::byte* data, std::size_t size, std::size_t alignment) noexcept = 0;
    virtual MemoryPoolStats stats() const noexcept = 0;
};

class BucketMemoryPool final : public RawMemoryPool {
public:
    explicit BucketMemoryPool(std::size_t reserve_bytes = 0, std::size_t blocks_per_slab = 64);
    ~BucketMemoryPool() override;

    BucketMemoryPool(const BucketMemoryPool&) = delete;
    BucketMemoryPool& operator=(const BucketMemoryPool&) = delete;

    Result<MemoryBlock> allocate(std::size_t size, std::size_t alignment = alignof(std::max_align_t)) override;
    void release(std::byte* data, std::size_t size, std::size_t alignment) noexcept override;
    MemoryPoolStats stats() const noexcept override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace core
