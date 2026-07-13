#pragma once

#include "logger_adapter.h"
#include "result.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ipc::media {

inline constexpr std::size_t kSharedFrameSessionIdCapacity = 128;
inline constexpr std::size_t kSharedFrameTraceIdCapacity = 128;
inline constexpr std::size_t kSharedFrameExecutionIdCapacity = 128;

enum class SharedFrameFormat : std::uint32_t {
    Unknown = 0,
    Jpeg = 1,
    Png = 2,
    Rgb = 3,
    Bgr = 4,
    Nv12 = 5,
    I420 = 6
};

struct SharedFrameMetadata {
    std::string execution_id;
    std::string session_id;
    std::string trace_id;
    std::uint64_t selected_sequence = 0;
    std::uint64_t transport_sequence = 0;
    std::uint64_t frame_id = 0;
    std::int64_t timestamp_us = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format = 0;
    double saliency = 0.0;
};

struct SharedFramePublishRequest {
    std::string_view execution_id;
    std::string_view session_id;
    std::string_view trace_id;
    std::uint64_t selected_sequence = 0;
    std::uint64_t transport_sequence = 0;
    std::uint64_t frame_id = 0;
    std::int64_t timestamp_us = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format = 0;
    double saliency = 0.0;
    std::span<const std::byte> payload;
};

/// 共享内存 region 配置；slot_count 和 payload_capacity 在 Create/Open 两端必须一致。
struct InferenceFrameSharedMemoryOptions {
    std::string name;
    std::size_t slot_count = 64;
    std::size_t payload_capacity = 4 * 1024 * 1024;
    std::optional<std::uint64_t> expected_epoch;
    bool remove_existing = false;
    bool remove_on_destroy = false;
};

struct InferenceFrameSharedMemorySnapshot {
    std::uint64_t epoch = 0;
    std::size_t slot_count = 0;
    std::size_t payload_capacity = 0;
    std::uint64_t published_frames = 0;
    std::uint64_t claimed_frames = 0;
    std::uint64_t acknowledged_frames = 0;
    std::uint64_t rejected_frames = 0;
    bool fenced = false;
    bool shutdown = false;
};

class ClaimedSharedFrame {
public:
    ClaimedSharedFrame() = default;
    ~ClaimedSharedFrame();

    ClaimedSharedFrame(const ClaimedSharedFrame&) = delete;
    ClaimedSharedFrame& operator=(const ClaimedSharedFrame&) = delete;
    ClaimedSharedFrame(ClaimedSharedFrame&&) noexcept;
    ClaimedSharedFrame& operator=(ClaimedSharedFrame&&) noexcept;

    /// 返回 slot 元数据；仅在 claim 有效且未 Acknowledge 时可使用。
    const SharedFrameMetadata& metadata() const noexcept;
    /// 返回共享 payload 视图；不得跨 Acknowledge、fence、移动赋值或析构保存。
    std::span<const std::byte> payload() const noexcept;
    /// 释放当前 slot；可重复调用，但实际确认至多执行一次。
    core::Status Acknowledge() noexcept;
    bool valid() const noexcept;

private:
    class Impl;
    explicit ClaimedSharedFrame(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;

    friend class SharedMemoryInferenceFrameChannel;
};

class IInferenceFrameIpcSink {
public:
    virtual ~IInferenceFrameIpcSink() = default;

    /// 发布一帧；request 中的 string_view/span 只在调用期间借用。
    /// ring 满时返回 ResourceExhausted，不覆盖已发布或已 claim 的 slot。
    virtual core::Status Publish(const SharedFramePublishRequest& request) = 0;
    /// 标记通道停止接收新数据；调用应保持幂等。
    virtual void Shutdown() = 0;
    virtual InferenceFrameSharedMemorySnapshot Snapshot() const = 0;
};

class IInferenceFrameIpcSource {
public:
    virtual ~IInferenceFrameIpcSource() = default;

    /// 非阻塞 claim 下一帧；无数据时返回可识别的失败 Status。
    /// 成功结果必须及时 Acknowledge，析构会执行 RAII 兜底确认。
    virtual core::Result<ClaimedSharedFrame> TryClaim() = 0;
    virtual void Shutdown() = 0;
    virtual InferenceFrameSharedMemorySnapshot Snapshot() const = 0;
};

class SharedMemoryInferenceFrameChannel final
    : public IInferenceFrameIpcSink,
      public IInferenceFrameIpcSource {
public:
    class Impl;

    /// 创建并初始化共享 region。
    /// @param options region 名称、slot 布局和清理策略。
    /// @param logger 可选日志适配器。
    static core::Result<std::unique_ptr<SharedMemoryInferenceFrameChannel>> Create(
        InferenceFrameSharedMemoryOptions options,
        core::LoggerAdapter logger = {});
    /// 打开已有 region，并验证 magic/version/layout/epoch。
    static core::Result<std::unique_ptr<SharedMemoryInferenceFrameChannel>> Open(
        InferenceFrameSharedMemoryOptions options,
        core::LoggerAdapter logger = {});
    /// 按名称移除 region；调用方必须先协调 producer/consumer 生命周期。
    static core::Status Remove(std::string_view name);

    ~SharedMemoryInferenceFrameChannel() override;

    SharedMemoryInferenceFrameChannel(const SharedMemoryInferenceFrameChannel&) = delete;
    SharedMemoryInferenceFrameChannel& operator=(const SharedMemoryInferenceFrameChannel&) = delete;

    core::Status Publish(const SharedFramePublishRequest& request) override;
    core::Result<ClaimedSharedFrame> TryClaim() override;
    /// 原子撤销当前 epoch，使旧 reader 和尚未确认的 claim 立即失效。
    core::Status Fence();
    void Shutdown() override;
    InferenceFrameSharedMemorySnapshot Snapshot() const override;

private:
    explicit SharedMemoryInferenceFrameChannel(std::shared_ptr<Impl> impl) noexcept;

    std::shared_ptr<Impl> impl_;
};

} // namespace ipc::media
