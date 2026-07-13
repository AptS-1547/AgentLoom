#pragma once

#include "result.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace core {

struct OrderedBitmapWindowOptions {
    std::size_t capacity = 1024;
    std::uint64_t first_sequence = 1;
};

struct OrderedBitmapWindowSnapshot {
    std::size_t capacity = 0;
    std::size_t pending = 0;
    std::size_t ready = 0;
    std::size_t skipped = 0;
    std::uint64_t base_sequence = 1;
    std::uint64_t sealed_sequence = 0;
    bool sealed = false;
    bool drained = false;
};

template <typename T>
struct OrderedBitmapItem {
    std::uint64_t sequence = 0;
    std::optional<T> value;
    Status terminal_status = Status::Ok();

    bool skipped() const noexcept {
        return !value.has_value();
    }
};

template <typename T>
class IOrderedBitmapWindow {
public:
    virtual ~IOrderedBitmapWindow() = default;

    virtual Status Admit(std::uint64_t sequence, T value) = 0;
    virtual Status MarkSkipped(std::uint64_t sequence, Status reason) = 0;
    virtual Status Seal(std::uint64_t final_sequence) = 0;
    virtual Result<OrderedBitmapItem<T>> TryTake() = 0;
    virtual Result<OrderedBitmapItem<T>> WaitTake(std::chrono::milliseconds timeout) = 0;
    virtual OrderedBitmapWindowSnapshot Snapshot() const = 0;
};

template <typename T>
class OrderedBitmapWindow final : public IOrderedBitmapWindow<T> {
public:
    explicit OrderedBitmapWindow(OrderedBitmapWindowOptions options = {})
        : options_(options),
          slots_(options.capacity),
          occupied_bitmap_(WordCount(options.capacity)),
          terminal_bitmap_(WordCount(options.capacity)),
          skipped_bitmap_(WordCount(options.capacity)),
          base_sequence_(options.first_sequence) {}

    Status Admit(std::uint64_t sequence, T value) override {
        std::lock_guard lock(mutex_);
        auto slot = PrepareSlotLocked(sequence);
        if (!slot.ok()) {
            return slot.status();
        }
        auto& target = slots_[slot.value()];
        target.sequence = sequence;
        target.value.emplace(std::move(value));
        target.terminal_status = Status::Ok();
        SetBit(occupied_bitmap_, slot.value());
        SetBit(terminal_bitmap_, slot.value());
        ++pending_;
        ++ready_;
        ready_cv_.notify_all();
        return Status::Ok();
    }

    Status MarkSkipped(std::uint64_t sequence, Status reason) override {
        if (reason.ok()) {
            return Status::Error(ErrorCode::InvalidArgument, "skipped sequence requires a failure status");
        }
        std::lock_guard lock(mutex_);
        auto slot = PrepareSlotLocked(sequence);
        if (!slot.ok()) {
            return slot.status();
        }
        auto& target = slots_[slot.value()];
        target.sequence = sequence;
        target.value.reset();
        target.terminal_status = std::move(reason);
        SetBit(occupied_bitmap_, slot.value());
        SetBit(terminal_bitmap_, slot.value());
        SetBit(skipped_bitmap_, slot.value());
        ++pending_;
        ++skipped_;
        ready_cv_.notify_all();
        return Status::Ok();
    }

    Status Seal(std::uint64_t final_sequence) override {
        std::lock_guard lock(mutex_);
        if (sealed_) {
            return sealed_sequence_ == final_sequence
                ? Status::Ok()
                : Status::Error(ErrorCode::AlreadyExists, "ordered bitmap window is already sealed");
        }
        if (final_sequence + 1 < base_sequence_) {
            return Status::Error(ErrorCode::InvalidArgument, "seal sequence precedes drained window");
        }
        sealed_ = true;
        sealed_sequence_ = final_sequence;
        ready_cv_.notify_all();
        return Status::Ok();
    }

    Result<OrderedBitmapItem<T>> TryTake() override {
        std::lock_guard lock(mutex_);
        return TakeLocked();
    }

    Result<OrderedBitmapItem<T>> WaitTake(std::chrono::milliseconds timeout) override {
        std::unique_lock lock(mutex_);
        const auto ready = ready_cv_.wait_for(lock, timeout, [&] {
            return BaseReadyLocked() || DrainedLocked();
        });
        if (!ready) {
            return Status::Error(ErrorCode::Timeout, "ordered bitmap window wait timed out");
        }
        return TakeLocked();
    }

    OrderedBitmapWindowSnapshot Snapshot() const override {
        std::lock_guard lock(mutex_);
        return {
            .capacity = options_.capacity,
            .pending = pending_,
            .ready = ready_,
            .skipped = skipped_,
            .base_sequence = base_sequence_,
            .sealed_sequence = sealed_sequence_,
            .sealed = sealed_,
            .drained = DrainedLocked(),
        };
    }

private:
    struct Slot {
        std::uint64_t sequence = 0;
        std::optional<T> value;
        Status terminal_status = Status::Ok();
    };

    static std::size_t WordCount(std::size_t capacity) noexcept {
        return (capacity + 63) / 64;
    }

    static void SetBit(std::vector<std::uint64_t>& bitmap, std::size_t index) noexcept {
        bitmap[index / 64] |= std::uint64_t{1} << (index % 64);
    }

    static void ClearBit(std::vector<std::uint64_t>& bitmap, std::size_t index) noexcept {
        bitmap[index / 64] &= ~(std::uint64_t{1} << (index % 64));
    }

    static bool TestBit(const std::vector<std::uint64_t>& bitmap, std::size_t index) noexcept {
        return (bitmap[index / 64] & (std::uint64_t{1} << (index % 64))) != 0;
    }

    Result<std::size_t> PrepareSlotLocked(std::uint64_t sequence) const {
        if (options_.capacity == 0 || options_.first_sequence == 0) {
            return Status::Error(ErrorCode::FailedPrecondition, "ordered bitmap window options are invalid");
        }
        if (sequence < base_sequence_) {
            return Status::Error(ErrorCode::AlreadyExists, "ordered bitmap sequence is already drained");
        }
        if (sealed_ && sequence > sealed_sequence_) {
            return Status::Error(ErrorCode::FailedPrecondition, "ordered bitmap sequence exceeds seal");
        }
        if (sequence - base_sequence_ >= options_.capacity) {
            return Status::Error(ErrorCode::ResourceExhausted, "ordered bitmap sequence exceeds window capacity");
        }
        const auto index = static_cast<std::size_t>(sequence % options_.capacity);
        if (TestBit(occupied_bitmap_, index) || slots_[index].sequence == sequence) {
            return Status::Error(ErrorCode::AlreadyExists, "ordered bitmap sequence already exists");
        }
        return index;
    }

    bool BaseReadyLocked() const noexcept {
        if (options_.capacity == 0) {
            return false;
        }
        const auto index = static_cast<std::size_t>(base_sequence_ % options_.capacity);
        return TestBit(terminal_bitmap_, index) && slots_[index].sequence == base_sequence_;
    }

    bool DrainedLocked() const noexcept {
        return sealed_ && base_sequence_ > sealed_sequence_;
    }

    Result<OrderedBitmapItem<T>> TakeLocked() {
        if (!BaseReadyLocked()) {
            if (DrainedLocked()) {
                return Status::Error(ErrorCode::Cancelled, "ordered bitmap window is drained");
            }
            return Status::Error(ErrorCode::NotFound, "next ordered bitmap sequence is not ready");
        }

        const auto sequence = base_sequence_;
        const auto index = static_cast<std::size_t>(sequence % options_.capacity);
        auto& slot = slots_[index];
        OrderedBitmapItem<T> item{
            .sequence = sequence,
            .value = std::move(slot.value),
            .terminal_status = std::move(slot.terminal_status),
        };
        const bool was_skipped = TestBit(skipped_bitmap_, index);
        ClearBit(occupied_bitmap_, index);
        ClearBit(terminal_bitmap_, index);
        ClearBit(skipped_bitmap_, index);
        slot.sequence = 0;
        slot.value.reset();
        slot.terminal_status = Status::Ok();
        --pending_;
        if (was_skipped) {
            --skipped_;
        } else {
            --ready_;
        }
        ++base_sequence_;
        ready_cv_.notify_all();
        return item;
    }

    OrderedBitmapWindowOptions options_;
    std::vector<Slot> slots_;
    std::vector<std::uint64_t> occupied_bitmap_;
    std::vector<std::uint64_t> terminal_bitmap_;
    std::vector<std::uint64_t> skipped_bitmap_;
    mutable std::mutex mutex_;
    std::condition_variable ready_cv_;
    std::uint64_t base_sequence_ = 1;
    std::uint64_t sealed_sequence_ = 0;
    std::size_t pending_ = 0;
    std::size_t ready_ = 0;
    std::size_t skipped_ = 0;
    bool sealed_ = false;
};

} // namespace core
