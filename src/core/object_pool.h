#pragma once

#include "memory_pool.h"
#include "result.h"

#include <new>
#include <type_traits>
#include <utility>

namespace core {

template <typename T>
class ObjectPool;

template <typename T>
class PooledObject {
public:
    PooledObject() noexcept = default;
    PooledObject(PooledObject&& other) noexcept
        : ptr_(other.ptr_), owner_(other.owner_) {
        other.ptr_ = nullptr;
        other.owner_ = nullptr;
    }

    PooledObject& operator=(PooledObject&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            owner_ = other.owner_;
            other.ptr_ = nullptr;
            other.owner_ = nullptr;
        }
        return *this;
    }

    ~PooledObject() {
        reset();
    }

    PooledObject(const PooledObject&) = delete;
    PooledObject& operator=(const PooledObject&) = delete;

    T* get() noexcept {
        return ptr_;
    }

    const T* get() const noexcept {
        return ptr_;
    }

    T& operator*() noexcept {
        return *ptr_;
    }

    const T& operator*() const noexcept {
        return *ptr_;
    }

    T* operator->() noexcept {
        return ptr_;
    }

    const T* operator->() const noexcept {
        return ptr_;
    }

    explicit operator bool() const noexcept {
        return ptr_ != nullptr;
    }

    void reset() noexcept {
        if (ptr_ && owner_) {
            owner_->destroy(ptr_);
        }
        ptr_ = nullptr;
        owner_ = nullptr;
    }

private:
    friend class ObjectPool<T>;

    PooledObject(T* ptr, ObjectPool<T>* owner) noexcept
        : ptr_(ptr), owner_(owner) {}

    T* ptr_ = nullptr;
    ObjectPool<T>* owner_ = nullptr;
};

template <typename T>
class ObjectPool {
public:
    ObjectPool() = default;
    explicit ObjectPool(RawMemoryPool& memory_pool) noexcept
        : memory_pool_(&memory_pool) {}

    template <typename... Args>
    Result<PooledObject<T>> make(Args&&... args) {
        if (!memory_pool_) {
            return Status::Error(ErrorCode::InvalidArgument, "memory pool is null");
        }

        auto block_result = memory_pool_->allocate(sizeof(T), alignof(T));
        if (!block_result) {
            return block_result.status();
        }

        MemoryBlock block = std::move(block_result.value());
        T* ptr = nullptr;
        try {
            ptr = new (block.data()) T(std::forward<Args>(args)...);
        } catch (...) {
            block.reset();
            return Status::Error(ErrorCode::InternalError, "object construction failed");
        }

        (void)block.release();
        return PooledObject<T>(ptr, this);
    }

private:
    void destroy(T* ptr) noexcept {
        if (!ptr || !memory_pool_) {
            return;
        }

        ptr->~T();
        memory_pool_->release(reinterpret_cast<std::byte*>(ptr), sizeof(T), alignof(T));
    }

    RawMemoryPool* memory_pool_ = nullptr;

    friend class PooledObject<T>;
};

} // namespace core
