#pragma once

#include "inference_frame_backlog.h"
#include "logger_adapter.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace media::inference {

struct MappedSpoolByteBudgetSnapshot {
    std::size_t max_bytes = 0;
    std::size_t reserved_bytes = 0;
    std::size_t available_bytes = 0;
};

class IMappedSpoolByteBudget {
public:
    virtual ~IMappedSpoolByteBudget() = default;

    virtual core::Status TryReserve(std::size_t bytes) = 0;
    virtual core::Status Release(std::size_t bytes) = 0;
    virtual MappedSpoolByteBudgetSnapshot Snapshot() const = 0;
};

class MappedSpoolByteBudget final : public IMappedSpoolByteBudget {
public:
    static core::Result<std::shared_ptr<MappedSpoolByteBudget>> Create(std::size_t max_bytes);

    ~MappedSpoolByteBudget() override;

    MappedSpoolByteBudget(const MappedSpoolByteBudget&) = delete;
    MappedSpoolByteBudget& operator=(const MappedSpoolByteBudget&) = delete;

    core::Status TryReserve(std::size_t bytes) override;
    core::Status Release(std::size_t bytes) override;
    MappedSpoolByteBudgetSnapshot Snapshot() const override;

private:
    class Impl;
    explicit MappedSpoolByteBudget(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

struct MappedInferenceFrameSpoolOptions {
    std::filesystem::path root_directory;
    std::string execution_id;
    std::size_t segment_bytes = 64 * 1024 * 1024;
    std::size_t max_spool_bytes = 1024 * 1024 * 1024;
    std::shared_ptr<IMappedSpoolByteBudget> shared_byte_budget;
    bool flush_on_append = true;
    bool remove_on_destroy = true;
};

struct SpoolFrameMetadata {
    std::string execution_id;
    std::uint64_t selected_sequence = 0;
    InferenceFrameMetadata frame;
};

struct SpoolRecordHandle {
    std::uint64_t selected_sequence = 0;
    std::size_t segment_index = 0;
    std::size_t record_offset = 0;
    std::size_t payload_size = 0;
};

struct MappedInferenceFrameSpoolSnapshot {
    std::size_t segment_count = 0;
    std::size_t admitted_records = 0;
    std::size_t replayed_records = 0;
    std::size_t payload_bytes = 0;
    std::size_t allocated_bytes = 0;
    bool sealed = false;
    bool cleaned = false;
};

class MappedSpoolFrameLease {
public:
    class Impl;

    MappedSpoolFrameLease();
    explicit MappedSpoolFrameLease(std::unique_ptr<Impl> impl) noexcept;
    ~MappedSpoolFrameLease();

    MappedSpoolFrameLease(const MappedSpoolFrameLease&) = delete;
    MappedSpoolFrameLease& operator=(const MappedSpoolFrameLease&) = delete;
    MappedSpoolFrameLease(MappedSpoolFrameLease&&) noexcept;
    MappedSpoolFrameLease& operator=(MappedSpoolFrameLease&&) noexcept;

    const SpoolFrameMetadata& metadata() const noexcept;
    std::span<const std::byte> bytes() const noexcept;
    bool valid() const noexcept;

private:
    std::unique_ptr<Impl> impl_;

    friend class MappedInferenceFrameSpool;
};

class IInferenceFrameSpool {
public:
    virtual ~IInferenceFrameSpool() = default;

    virtual core::Result<SpoolRecordHandle> Append(
        const SpoolFrameMetadata& metadata,
        std::span<const std::byte> encoded_payload) = 0;
    virtual core::Status Seal() = 0;
    virtual core::Result<std::optional<MappedSpoolFrameLease>> ReplayNext() = 0;
    virtual MappedInferenceFrameSpoolSnapshot Snapshot() const = 0;
    virtual core::Status Cleanup() = 0;
};

class MappedInferenceFrameSpool final : public IInferenceFrameSpool {
public:
    static core::Result<std::unique_ptr<MappedInferenceFrameSpool>> Create(
        MappedInferenceFrameSpoolOptions options,
        core::LoggerAdapter logger = {});

    ~MappedInferenceFrameSpool() override;

    MappedInferenceFrameSpool(const MappedInferenceFrameSpool&) = delete;
    MappedInferenceFrameSpool& operator=(const MappedInferenceFrameSpool&) = delete;
    MappedInferenceFrameSpool(MappedInferenceFrameSpool&&) = delete;
    MappedInferenceFrameSpool& operator=(MappedInferenceFrameSpool&&) = delete;

    core::Result<SpoolRecordHandle> Append(
        const SpoolFrameMetadata& metadata,
        std::span<const std::byte> encoded_payload) override;
    core::Status Seal() override;
    core::Result<std::optional<MappedSpoolFrameLease>> ReplayNext() override;
    MappedInferenceFrameSpoolSnapshot Snapshot() const override;
    core::Status Cleanup() override;

private:
    class Impl;
    explicit MappedInferenceFrameSpool(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace media::inference
