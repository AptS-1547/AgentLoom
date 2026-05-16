#pragma once

#include "memory_pool.h"
#include "result.h"
#include "shared_memory_block.h"

#include <cstddef>
#include <cstring>
#include <string_view>
#include <utility>

namespace net {

class SharedBuffer {
public:
    SharedBuffer() noexcept = default;

    static core::Result<SharedBuffer> Allocate(core::RawMemoryPool& pool,
                                               std::size_t size,
                                               std::size_t alignment = alignof(std::max_align_t)) {
        return AllocateCapacity(pool, size, size, alignment);
    }

    static core::Result<SharedBuffer> AllocateCapacity(core::RawMemoryPool& pool,
                                                       std::size_t capacity,
                                                       std::size_t initial_size = 0,
                                                       std::size_t alignment = alignof(std::max_align_t)) {
        if (initial_size > capacity) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "buffer capacity exceeded");
        }
        if (capacity == 0) {
            return SharedBuffer();
        }

        auto block_result = core::SharedMemoryBlock::allocate(pool, capacity, alignment);
        if (!block_result.ok()) {
            return block_result.status();
        }
        return SharedBuffer(std::move(block_result).value(), capacity, initial_size);
    }

    static core::Result<SharedBuffer> Copy(core::RawMemoryPool& pool,
                                           std::string_view data,
                                           std::size_t alignment = alignof(std::max_align_t)) {
        return SafeCopy(pool, data, data.size(), alignment);
    }

    static core::Result<SharedBuffer> SafeCopy(core::RawMemoryPool& pool,
                                               std::string_view data,
                                               std::size_t max_bytes,
                                               std::size_t alignment = alignof(std::max_align_t)) {
        if (data.size() > max_bytes) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "buffer capacity exceeded");
        }

        auto buffer_result = AllocateCapacity(pool, data.size(), 0, alignment);
        if (!buffer_result.ok()) {
            return buffer_result.status();
        }

        auto buffer = std::move(buffer_result).value();
        auto status = buffer.Assign(data);
        if (!status.ok()) {
            return status;
        }
        return buffer;
    }

    std::byte* data() noexcept {
        return block_.data();
    }

    const std::byte* data() const noexcept {
        return block_.data();
    }

    char* char_data() noexcept {
        return reinterpret_cast<char*>(data());
    }

    const char* char_data() const noexcept {
        return reinterpret_cast<const char*>(data());
    }

    std::size_t size() const noexcept {
        return size_;
    }

    std::size_t capacity() const noexcept {
        return capacity_;
    }

    bool empty() const noexcept {
        return size_ == 0;
    }

    bool valid() const noexcept {
        return block_.valid();
    }

    explicit operator bool() const noexcept {
        return valid();
    }

    std::string_view view() const noexcept {
        if (size_ == 0) {
            return {};
        }
        return std::string_view(char_data(), size_);
    }

    core::Status resize(std::size_t size) noexcept {
        if (size > block_.size()) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "buffer capacity exceeded");
        }
        if (size > capacity_) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "buffer capacity exceeded");
        }
        size_ = size;
        return core::Status::Ok();
    }

    core::Status Write(std::size_t offset, const void* src, std::size_t bytes) noexcept {
        if (bytes > 0 && src == nullptr) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "source buffer is null");
        }
        const auto current_capacity = capacity();
        if (offset > current_capacity || bytes > current_capacity - offset) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "buffer capacity exceeded");
        }
        if (bytes > 0) {
            std::memcpy(data() + offset, src, bytes);
        }
        const auto next_size = offset + bytes;
        if (next_size > size_) {
            size_ = next_size;
        }
        return core::Status::Ok();
    }

    core::Status Assign(std::string_view data) noexcept {
        if (data.size() > capacity()) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "buffer capacity exceeded");
        }
        if (!data.empty() && data.data() == nullptr) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "source buffer is null");
        }
        size_ = 0;
        return Write(0, data.data(), data.size());
    }

    core::SharedMemoryBlock& block() noexcept {
        return block_;
    }

    const core::SharedMemoryBlock& block() const noexcept {
        return block_;
    }

    core::SharedMemoryBlock take_block() noexcept {
        size_ = 0;
        capacity_ = 0;
        return std::move(block_);
    }

    void reset() noexcept {
        block_.reset();
        size_ = 0;
        capacity_ = 0;
    }

private:
    SharedBuffer(core::SharedMemoryBlock block, std::size_t size) noexcept
        : block_(std::move(block)), capacity_(size), size_(size) {}

    SharedBuffer(core::SharedMemoryBlock block, std::size_t capacity, std::size_t size) noexcept
        : block_(std::move(block)), capacity_(capacity), size_(size) {}

    core::SharedMemoryBlock block_;
    std::size_t capacity_ = 0;
    std::size_t size_ = 0;
};

} // namespace net
