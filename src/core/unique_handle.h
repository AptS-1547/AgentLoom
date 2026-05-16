#pragma once

#include <utility>

namespace core {

struct adopt_handle_t {
    explicit adopt_handle_t() = default;
};

inline constexpr adopt_handle_t adopt_handle{};

template <typename Resource>
struct ResourceTraits;

template <typename Resource>
class UniqueHandle {
public:
    using traits_type = ResourceTraits<Resource>;
    using handle_type = typename traits_type::handle_type;

    UniqueHandle() noexcept = default;
    UniqueHandle(adopt_handle_t, handle_type handle) noexcept
        : handle_(handle) {}

    ~UniqueHandle() noexcept {
        reset();
    }

    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;

    UniqueHandle(UniqueHandle&& other) noexcept
        : handle_(other.release()) {}

    UniqueHandle& operator=(UniqueHandle&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    bool valid() const noexcept {
        return traits_type::valid(handle_);
    }

    explicit operator bool() const noexcept {
        return valid();
    }

    handle_type& get() noexcept {
        return handle_;
    }

    const handle_type& get() const noexcept {
        return handle_;
    }

    handle_type release() noexcept {
        handle_type old = handle_;
        handle_ = traits_type::invalid();
        return old;
    }

    void reset(handle_type handle = traits_type::invalid()) noexcept {
        if (traits_type::valid(handle_)) {
            traits_type::close(handle_);
        }
        handle_ = handle;
    }

private:
    handle_type handle_ = traits_type::invalid();
};

template <typename Resource>
UniqueHandle<Resource> make_unique_handle(typename ResourceTraits<Resource>::handle_type handle) noexcept {
    return UniqueHandle<Resource>(adopt_handle, handle);
}

} // namespace core
