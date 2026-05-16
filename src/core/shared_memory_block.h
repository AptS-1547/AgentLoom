#pragma once

#include "memory_pool.h"

#include <atomic>
#include <cstddef>
#include <new>
#include <utility>

namespace core {

class SharedMemoryBlock {
public:
    SharedMemoryBlock() noexcept = default;
    SharedMemoryBlock(const SharedMemoryBlock& other) noexcept
        : control_(other.control_) {
        retain();
    }

    SharedMemoryBlock& operator=(const SharedMemoryBlock& other) noexcept {
        if (this != &other) {
            reset();
            control_ = other.control_;
            retain();
        }
        return *this;
    }

    SharedMemoryBlock(SharedMemoryBlock&& other) noexcept
        : control_(other.control_) {
        other.control_ = nullptr;
    }

    SharedMemoryBlock& operator=(SharedMemoryBlock&& other) noexcept {
        if (this != &other) {
            reset();
            control_ = other.control_;
            other.control_ = nullptr;
        }
        return *this;
    }

    ~SharedMemoryBlock() {
        reset();
    }

    static Result<SharedMemoryBlock> allocate(RawMemoryPool& pool, std::size_t size, std::size_t alignment = alignof(std::max_align_t)) {
        auto block_result = pool.allocate(size, alignment);
        if (!block_result) {
            return block_result.status();
        }
        return adopt(std::move(block_result.value()));
    }

    static Result<SharedMemoryBlock> adopt(MemoryBlock&& block) {
        auto* owner = block.owner();
        auto* data = block.data();
        auto size = block.size();
        auto alignment = block.alignment();
        if (!owner || !data) {
            return Status::Error(ErrorCode::InvalidArgument, "memory block is empty");
        }

        ControlBlock* control = nullptr;
        try {
            control = new ControlBlock;
        } catch (const std::bad_alloc&) {
            return Status::Error(ErrorCode::OutOfMemory, "control block allocation failed");
        }

        control->owner = owner;
        control->data = block.release();
        control->size = size;
        control->alignment = alignment;
        return SharedMemoryBlock(control);
    }

    std::byte* data() noexcept {
        return control_ ? control_->data : nullptr;
    }

    const std::byte* data() const noexcept {
        return control_ ? control_->data : nullptr;
    }

    std::size_t size() const noexcept {
        return control_ ? control_->size : 0;
    }

    std::size_t use_count() const noexcept {
        return control_ ? control_->refs.load(std::memory_order_acquire) : 0;
    }

    bool valid() const noexcept {
        return control_ && control_->data;
    }

    explicit operator bool() const noexcept {
        return valid();
    }

    void reset() noexcept {
        if (!control_) {
            return;
        }
        if (control_->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            if (control_->owner && control_->data) {
                control_->owner->release(control_->data, control_->size, control_->alignment);
            }
            delete control_;
        }
        control_ = nullptr;
    }

private:
    struct ControlBlock {
        std::atomic<std::size_t> refs{1};
        RawMemoryPool* owner = nullptr;
        std::byte* data = nullptr;
        std::size_t size = 0;
        std::size_t alignment = 0;
    };

    explicit SharedMemoryBlock(ControlBlock* control) noexcept
        : control_(control) {}

    void retain() noexcept {
        if (control_) {
            control_->refs.fetch_add(1, std::memory_order_relaxed);
        }
    }

    ControlBlock* control_ = nullptr;
};

} // namespace core
