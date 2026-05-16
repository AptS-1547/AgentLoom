#include "memory_pool.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <new>
#include <vector>

namespace core {

namespace {

constexpr std::size_t kDefaultBlockSize = 64;
constexpr std::size_t kMinBlocksPerSlab = 8;

std::size_t RoundUp(std::size_t value, std::size_t alignment) noexcept {
    if (alignment <= 1) {
        return value;
    }
    const auto remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

std::size_t NormalizeAlignment(std::size_t alignment) noexcept {
    return std::max(alignment, alignof(std::max_align_t));
}

} // namespace

MemoryBlock::MemoryBlock(RawMemoryPool* owner, std::byte* data, std::size_t size, std::size_t alignment) noexcept
    : owner_(owner), data_(data), size_(size), alignment_(alignment) {}

MemoryBlock::MemoryBlock(MemoryBlock&& other) noexcept
    : owner_(other.owner_), data_(other.data_), size_(other.size_), alignment_(other.alignment_) {
    other.owner_ = nullptr;
    other.data_ = nullptr;
    other.size_ = 0;
    other.alignment_ = 0;
}

MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) noexcept {
    if (this != &other) {
        reset();
        owner_ = other.owner_;
        data_ = other.data_;
        size_ = other.size_;
        alignment_ = other.alignment_;
        other.owner_ = nullptr;
        other.data_ = nullptr;
        other.size_ = 0;
        other.alignment_ = 0;
    }
    return *this;
}

MemoryBlock::~MemoryBlock() {
    reset();
}

std::byte* MemoryBlock::data() noexcept {
    return data_;
}

const std::byte* MemoryBlock::data() const noexcept {
    return data_;
}

std::size_t MemoryBlock::size() const noexcept {
    return size_;
}

std::size_t MemoryBlock::alignment() const noexcept {
    return alignment_;
}

RawMemoryPool* MemoryBlock::owner() const noexcept {
    return owner_;
}

bool MemoryBlock::empty() const noexcept {
    return data_ == nullptr;
}

std::byte* MemoryBlock::release() noexcept {
    auto* data = data_;
    owner_ = nullptr;
    data_ = nullptr;
    size_ = 0;
    alignment_ = 0;
    return data;
}

void MemoryBlock::reset() noexcept {
    if (owner_ && data_) {
        owner_->release(data_, size_, alignment_);
    }
    owner_ = nullptr;
    data_ = nullptr;
    size_ = 0;
    alignment_ = 0;
}

class BucketMemoryPool::Impl {
public:
    Impl(std::size_t reserve_bytes, std::size_t blocks_per_slab)
        : reserve_bytes_(reserve_bytes),
          blocks_per_slab_(std::max(blocks_per_slab, kMinBlocksPerSlab)) {}

    ~Impl() {
        for (auto& slab : slabs_) {
            ::operator delete(slab.data, std::align_val_t(slab.alignment));
        }
    }

    Result<MemoryBlock> allocate(BucketMemoryPool& owner, std::size_t size, std::size_t alignment) {
        if (size == 0) {
            return Status::Error(ErrorCode::InvalidArgument, "size is zero");
        }

        alignment = NormalizeAlignment(alignment);
        const auto block_size = NormalizeSize(size, alignment);

        std::lock_guard lock(mutex_);
        auto& bucket = EnsureBucket(block_size, alignment);
        if (!bucket.free_list) {
            auto status = RefillBucket(bucket);
            if (!status.ok()) {
                return status;
            }
        }

        auto* node = bucket.free_list;
        bucket.free_list = node->next;
        --bucket.free_count;

        allocated_bytes_ += bucket.block_size;
        if (free_bytes_ >= bucket.block_size) {
            free_bytes_ -= bucket.block_size;
        } else {
            free_bytes_ = 0;
        }
        ++allocations_;
        peak_allocated_bytes_ = std::max(peak_allocated_bytes_, allocated_bytes_);

        return MemoryBlock(&owner, reinterpret_cast<std::byte*>(node), bucket.block_size, bucket.alignment);
    }

    void release(std::byte* data, std::size_t size, std::size_t alignment) noexcept {
        if (!data) {
            return;
        }

        alignment = NormalizeAlignment(alignment);
        const auto block_size = NormalizeSize(size, alignment);

        std::lock_guard lock(mutex_);
        Bucket* bucket = FindBucket(block_size, alignment);
        if (!bucket) {
            return;
        }

        auto* node = reinterpret_cast<FreeNode*>(data);
        node->next = bucket->free_list;
        bucket->free_list = node;
        ++bucket->free_count;

        if (allocated_bytes_ >= bucket->block_size) {
            allocated_bytes_ -= bucket->block_size;
        } else {
            allocated_bytes_ = 0;
        }
        free_bytes_ += bucket->block_size;
        ++deallocations_;
    }

    MemoryPoolStats stats() const noexcept {
        std::lock_guard lock(mutex_);
        MemoryPoolStats result;
        result.allocated_bytes = allocated_bytes_;
        result.free_bytes = free_bytes_;
        result.allocations = allocations_;
        result.deallocations = deallocations_;
        result.peak_allocated_bytes = peak_allocated_bytes_;
        return result;
    }

private:
    struct FreeNode {
        FreeNode* next = nullptr;
    };

    struct Slab {
        std::byte* data = nullptr;
        std::size_t size = 0;
        std::size_t alignment = 0;
    };

    struct Bucket {
        std::size_t block_size = 0;
        std::size_t alignment = 0;
        FreeNode* free_list = nullptr;
        std::size_t free_count = 0;
    };

    static std::size_t NormalizeSize(std::size_t size, std::size_t alignment) noexcept {
        return std::max(RoundUp(size, alignment), kDefaultBlockSize);
    }

    Bucket* FindBucket(std::size_t block_size, std::size_t alignment) noexcept {
        for (auto& bucket : buckets_) {
            if (bucket.block_size == block_size && bucket.alignment == alignment) {
                return &bucket;
            }
        }
        return nullptr;
    }

    Bucket& EnsureBucket(std::size_t block_size, std::size_t alignment) {
        if (auto* bucket = FindBucket(block_size, alignment)) {
            return *bucket;
        }

        buckets_.push_back(Bucket{block_size, alignment, nullptr, 0});
        return buckets_.back();
    }

    Status RefillBucket(Bucket& bucket) noexcept {
        const auto desired_bytes = reserve_bytes_ > 0 ? std::max(reserve_bytes_, bucket.block_size) : bucket.block_size * blocks_per_slab_;
        const auto block_count = std::max<std::size_t>(1, desired_bytes / bucket.block_size);
        const auto slab_size = bucket.block_size * block_count;

        std::byte* data = nullptr;
        try {
            data = static_cast<std::byte*>(::operator new(slab_size, std::align_val_t(bucket.alignment)));
        } catch (const std::bad_alloc&) {
            return Status::Error(ErrorCode::OutOfMemory, "slab allocation failed");
        }

        slabs_.push_back(Slab{data, slab_size, bucket.alignment});
        for (std::size_t i = 0; i < block_count; ++i) {
            auto* node = reinterpret_cast<FreeNode*>(data + i * bucket.block_size);
            node->next = bucket.free_list;
            bucket.free_list = node;
            ++bucket.free_count;
        }
        free_bytes_ += slab_size;
        return Status::Ok();
    }

    std::size_t reserve_bytes_ = 0;
    std::size_t blocks_per_slab_ = 64;
    mutable std::mutex mutex_;
    std::vector<Bucket> buckets_;
    std::vector<Slab> slabs_;
    std::size_t allocated_bytes_ = 0;
    std::size_t free_bytes_ = 0;
    std::size_t allocations_ = 0;
    std::size_t deallocations_ = 0;
    std::size_t peak_allocated_bytes_ = 0;
};

BucketMemoryPool::BucketMemoryPool(std::size_t reserve_bytes, std::size_t blocks_per_slab)
    : impl_(std::make_unique<Impl>(reserve_bytes, blocks_per_slab)) {}

BucketMemoryPool::~BucketMemoryPool() = default;

Result<MemoryBlock> BucketMemoryPool::allocate(std::size_t size, std::size_t alignment) {
    return impl_->allocate(*this, size, alignment);
}

void BucketMemoryPool::release(std::byte* data, std::size_t size, std::size_t alignment) noexcept {
    impl_->release(data, size, alignment);
}

MemoryPoolStats BucketMemoryPool::stats() const noexcept {
    return impl_->stats();
}

} // namespace core
