#include "inference_frame_shared_memory.h"

#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <utility>

namespace ipc::media {
namespace {

namespace bip = boost::interprocess;

constexpr std::uint64_t kRegionMagic = 0x3143504952464941ULL;
constexpr std::uint32_t kRegionVersion = 1;
constexpr std::uint32_t kRegionInitializing = 1;
constexpr std::uint32_t kRegionReady = 2;
constexpr std::size_t kCacheLineSize = 64;

static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
static_assert(std::atomic<std::uint32_t>::is_always_lock_free);

struct alignas(kCacheLineSize) RegionHeader {
    std::uint64_t magic = kRegionMagic;
    std::uint32_t version = kRegionVersion;
    std::uint32_t header_size = 0;
    std::uint64_t region_size = 0;
    std::uint64_t slot_headers_offset = 0;
    std::uint64_t payload_offset = 0;
    std::uint32_t slot_count = 0;
    std::uint32_t payload_capacity = 0;
    std::atomic<std::uint32_t> state{kRegionInitializing};
    std::uint32_t reserved = 0;
    std::atomic<std::uint64_t> epoch{0};
    alignas(kCacheLineSize) std::atomic<std::uint64_t> enqueue_position{0};
    alignas(kCacheLineSize) std::atomic<std::uint64_t> dequeue_position{0};
    alignas(kCacheLineSize) std::atomic<std::uint64_t> published_frames{0};
    std::atomic<std::uint64_t> claimed_frames{0};
    std::atomic<std::uint64_t> acknowledged_frames{0};
    std::atomic<std::uint64_t> rejected_frames{0};
};

struct alignas(kCacheLineSize) SlotHeader {
    std::atomic<std::uint64_t> sequence{0};
    std::uint64_t epoch = 0;
    std::uint64_t frame_id = 0;
    std::int64_t timestamp_us = 0;
    std::uint32_t payload_size = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format = 0;
    std::uint16_t session_id_size = 0;
    std::uint16_t trace_id_size = 0;
    std::uint32_t reserved = 0;
    double saliency = 0.0;
    char session_id[kSharedFrameSessionIdCapacity]{};
    char trace_id[kSharedFrameTraceIdCapacity]{};
};

struct RegionLayout {
    std::size_t slot_headers_offset = 0;
    std::size_t payload_offset = 0;
    std::size_t region_size = 0;
};

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

core::Result<RegionLayout> CalculateLayout(std::size_t slot_count, std::size_t payload_capacity) {
    if (slot_count < 2 || slot_count > std::numeric_limits<std::uint32_t>::max()) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "shared frame slot_count must be at least 2");
    }
    if (payload_capacity == 0 || payload_capacity > std::numeric_limits<std::uint32_t>::max()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "shared frame payload_capacity is invalid");
    }

    const auto slot_headers_offset = AlignUp(sizeof(RegionHeader), alignof(SlotHeader));
    if (slot_count > (std::numeric_limits<std::size_t>::max() - slot_headers_offset) / sizeof(SlotHeader)) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "shared frame slot layout overflows");
    }
    const auto payload_offset = AlignUp(slot_headers_offset + slot_count * sizeof(SlotHeader), kCacheLineSize);
    if (slot_count > (std::numeric_limits<std::size_t>::max() - payload_offset) / payload_capacity) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "shared frame payload layout overflows");
    }
    return RegionLayout{
        .slot_headers_offset = slot_headers_offset,
        .payload_offset = payload_offset,
        .region_size = payload_offset + slot_count * payload_capacity,
    };
}

std::uint64_t GenerateEpoch() noexcept {
    static std::atomic<std::uint64_t> counter{1};
    const auto now = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return now ^ counter.fetch_add(1, std::memory_order_relaxed);
}

core::Status ValidatePublishRequest(
    const SharedFramePublishRequest& request,
    std::size_t payload_capacity) {
    if (request.session_id.empty() || request.session_id.size() >= kSharedFrameSessionIdCapacity) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "shared frame session_id is invalid");
    }
    if (request.trace_id.size() >= kSharedFrameTraceIdCapacity) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "shared frame trace_id is too long");
    }
    if (request.payload.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "shared frame payload is empty");
    }
    if (request.payload.size() > payload_capacity) {
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "shared frame payload exceeds slot capacity");
    }
    return core::Status::Ok();
}

core::Status InterprocessFailure(
    const core::LoggerAdapter& logger,
    std::string_view operation,
    const bip::interprocess_exception& exception) {
    logger.error(
        "[frame-ipc] {} failed code={} detail={}",
        operation,
        static_cast<int>(exception.get_error_code()),
        exception.what());
    return core::Status::Error(
        core::ErrorCode::Unavailable,
        std::string(operation) + " failed");
}

} // namespace

class SharedMemoryInferenceFrameChannel::Impl
    : public std::enable_shared_from_this<SharedMemoryInferenceFrameChannel::Impl> {
public:
    Impl(
        InferenceFrameSharedMemoryOptions options,
        bip::shared_memory_object object,
        bip::mapped_region region,
        core::LoggerAdapter logger)
        : options_(std::move(options)),
          object_(std::move(object)),
          region_(std::move(region)),
          logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("frame-ipc")) {
        header_ = static_cast<RegionHeader*>(region_.get_address());
        slots_ = reinterpret_cast<SlotHeader*>(
            static_cast<std::byte*>(region_.get_address()) + header_->slot_headers_offset);
        payload_base_ = static_cast<std::byte*>(region_.get_address()) + header_->payload_offset;
    }

    ~Impl() {
        if (options_.remove_on_destroy) {
            bip::shared_memory_object::remove(options_.name.c_str());
        }
    }

    core::Status Publish(const SharedFramePublishRequest& request) {
        if (shutdown_.load(std::memory_order_acquire)) {
            return Reject(core::Status::Error(core::ErrorCode::Cancelled, "shared frame channel is shut down"));
        }
        const auto validation = ValidatePublishRequest(request, header_->payload_capacity);
        if (!validation.ok()) {
            return Reject(validation);
        }

        std::uint64_t position = header_->enqueue_position.load(std::memory_order_relaxed);
        SlotHeader* slot = nullptr;
        for (;;) {
            auto& candidate = slots_[position % header_->slot_count];
            const auto sequence = candidate.sequence.load(std::memory_order_acquire);
            const auto difference = static_cast<std::int64_t>(sequence) -
                                    static_cast<std::int64_t>(position);
            if (difference == 0) {
                if (header_->enqueue_position.compare_exchange_weak(
                        position,
                        position + 1,
                        std::memory_order_relaxed)) {
                    slot = &candidate;
                    break;
                }
                continue;
            }
            if (difference < 0) {
                return Reject(core::Status::Error(
                    core::ErrorCode::ResourceExhausted,
                    "shared frame channel is full"));
            }
            position = header_->enqueue_position.load(std::memory_order_relaxed);
            std::this_thread::yield();
        }

        const auto slot_index = static_cast<std::size_t>(position % header_->slot_count);
        auto* payload = Payload(slot_index);
        std::memcpy(payload, request.payload.data(), request.payload.size());

        slot->epoch = header_->epoch.load(std::memory_order_acquire);
        slot->frame_id = request.frame_id;
        slot->timestamp_us = request.timestamp_us;
        slot->payload_size = static_cast<std::uint32_t>(request.payload.size());
        slot->width = request.width;
        slot->height = request.height;
        slot->format = request.format;
        slot->saliency = request.saliency;
        slot->session_id_size = static_cast<std::uint16_t>(request.session_id.size());
        slot->trace_id_size = static_cast<std::uint16_t>(request.trace_id.size());
        std::memcpy(slot->session_id, request.session_id.data(), request.session_id.size());
        slot->session_id[request.session_id.size()] = '\0';
        if (!request.trace_id.empty()) {
            std::memcpy(slot->trace_id, request.trace_id.data(), request.trace_id.size());
        }
        slot->trace_id[request.trace_id.size()] = '\0';

        slot->sequence.store(position + 1, std::memory_order_release);
        header_->published_frames.fetch_add(1, std::memory_order_relaxed);
        return core::Status::Ok();
    }

    core::Result<ClaimedSharedFrame> TryClaim();

    void Shutdown() noexcept {
        shutdown_.store(true, std::memory_order_release);
    }

    InferenceFrameSharedMemorySnapshot Snapshot() const {
        return {
            .epoch = header_->epoch.load(std::memory_order_acquire),
            .slot_count = header_->slot_count,
            .payload_capacity = header_->payload_capacity,
            .published_frames = header_->published_frames.load(std::memory_order_relaxed),
            .claimed_frames = header_->claimed_frames.load(std::memory_order_relaxed),
            .acknowledged_frames = header_->acknowledged_frames.load(std::memory_order_relaxed),
            .rejected_frames = header_->rejected_frames.load(std::memory_order_relaxed),
            .shutdown = shutdown_.load(std::memory_order_acquire),
        };
    }

    std::byte* Payload(std::size_t slot_index) const noexcept {
        return payload_base_ + slot_index * header_->payload_capacity;
    }

    core::Status Reject(core::Status status) {
        header_->rejected_frames.fetch_add(1, std::memory_order_relaxed);
        if (status.code() == core::ErrorCode::ResourceExhausted) {
            logger_.debug(
                "[frame-ipc] backpressure code={} message={}",
                static_cast<int>(status.code()),
                status.message());
        } else {
            logger_.warn(
                "[frame-ipc] request rejected code={} message={}",
                static_cast<int>(status.code()),
                status.message());
        }
        return status;
    }

    InferenceFrameSharedMemoryOptions options_;
    bip::shared_memory_object object_;
    bip::mapped_region region_;
    core::LoggerAdapter logger_;
    RegionHeader* header_ = nullptr;
    SlotHeader* slots_ = nullptr;
    std::byte* payload_base_ = nullptr;
    std::atomic<bool> shutdown_{false};
};

class ClaimedSharedFrame::Impl {
public:
    Impl(
        std::shared_ptr<SharedMemoryInferenceFrameChannel::Impl> channel,
        SlotHeader* slot,
        std::uint64_t position,
        SharedFrameMetadata metadata,
        std::span<const std::byte> payload)
        : channel_(std::move(channel)),
          slot_(slot),
          position_(position),
          metadata_(std::move(metadata)),
          payload_(payload) {}

    ~Impl() {
        Acknowledge();
    }

    core::Status Acknowledge() noexcept {
        if (acknowledged_.exchange(true, std::memory_order_acq_rel)) {
            return core::Status::Ok();
        }
        slot_->sequence.store(
            position_ + channel_->header_->slot_count,
            std::memory_order_release);
        channel_->header_->acknowledged_frames.fetch_add(1, std::memory_order_relaxed);
        payload_ = {};
        return core::Status::Ok();
    }

    std::shared_ptr<SharedMemoryInferenceFrameChannel::Impl> channel_;
    SlotHeader* slot_ = nullptr;
    std::uint64_t position_ = 0;
    SharedFrameMetadata metadata_;
    std::span<const std::byte> payload_;
    std::atomic<bool> acknowledged_{false};
};

core::Result<ClaimedSharedFrame> SharedMemoryInferenceFrameChannel::Impl::TryClaim() {
    if (shutdown_.load(std::memory_order_acquire)) {
        return core::Status::Error(core::ErrorCode::Cancelled, "shared frame channel is shut down");
    }

    std::uint64_t position = header_->dequeue_position.load(std::memory_order_relaxed);
    SlotHeader* slot = nullptr;
    for (;;) {
        auto& candidate = slots_[position % header_->slot_count];
        const auto sequence = candidate.sequence.load(std::memory_order_acquire);
        const auto difference = static_cast<std::int64_t>(sequence) -
                                static_cast<std::int64_t>(position + 1);
        if (difference == 0) {
            if (header_->dequeue_position.compare_exchange_weak(
                    position,
                    position + 1,
                    std::memory_order_relaxed)) {
                slot = &candidate;
                break;
            }
            continue;
        }
        if (difference < 0) {
            return core::Status::Error(core::ErrorCode::NotFound, "shared frame channel is empty");
        }
        position = header_->dequeue_position.load(std::memory_order_relaxed);
        std::this_thread::yield();
    }

    const auto epoch = header_->epoch.load(std::memory_order_acquire);
    const auto valid = slot->epoch == epoch &&
                       slot->payload_size > 0 &&
                       slot->payload_size <= header_->payload_capacity &&
                       slot->session_id_size > 0 &&
                       slot->session_id_size < kSharedFrameSessionIdCapacity &&
                       slot->trace_id_size < kSharedFrameTraceIdCapacity;
    if (!valid) {
        slot->sequence.store(position + header_->slot_count, std::memory_order_release);
        header_->acknowledged_frames.fetch_add(1, std::memory_order_relaxed);
        return Reject(core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "shared frame slot metadata is inconsistent"));
    }

    SharedFrameMetadata metadata;
    metadata.session_id.assign(slot->session_id, slot->session_id_size);
    metadata.trace_id.assign(slot->trace_id, slot->trace_id_size);
    metadata.frame_id = slot->frame_id;
    metadata.timestamp_us = slot->timestamp_us;
    metadata.width = slot->width;
    metadata.height = slot->height;
    metadata.format = slot->format;
    metadata.saliency = slot->saliency;

    const auto slot_index = static_cast<std::size_t>(position % header_->slot_count);
    auto claim = ClaimedSharedFrame(std::make_unique<ClaimedSharedFrame::Impl>(
        shared_from_this(),
        slot,
        position,
        std::move(metadata),
        std::span<const std::byte>(Payload(slot_index), slot->payload_size)));
    header_->claimed_frames.fetch_add(1, std::memory_order_relaxed);
    return claim;
}

ClaimedSharedFrame::ClaimedSharedFrame(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ClaimedSharedFrame::~ClaimedSharedFrame() = default;
ClaimedSharedFrame::ClaimedSharedFrame(ClaimedSharedFrame&&) noexcept = default;
ClaimedSharedFrame& ClaimedSharedFrame::operator=(ClaimedSharedFrame&&) noexcept = default;

const SharedFrameMetadata& ClaimedSharedFrame::metadata() const noexcept {
    static const SharedFrameMetadata empty;
    return impl_ ? impl_->metadata_ : empty;
}

std::span<const std::byte> ClaimedSharedFrame::payload() const noexcept {
    return impl_ ? impl_->payload_ : std::span<const std::byte>{};
}

core::Status ClaimedSharedFrame::Acknowledge() noexcept {
    return impl_ ? impl_->Acknowledge() : core::Status::Ok();
}

bool ClaimedSharedFrame::valid() const noexcept {
    return impl_ && !impl_->acknowledged_.load(std::memory_order_acquire) && !impl_->payload_.empty();
}

core::Result<std::unique_ptr<SharedMemoryInferenceFrameChannel>>
SharedMemoryInferenceFrameChannel::Create(
    InferenceFrameSharedMemoryOptions options,
    core::LoggerAdapter logger) {
    if (options.name.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "shared frame channel name is required");
    }
    auto layout = CalculateLayout(options.slot_count, options.payload_capacity);
    if (!layout.ok()) {
        return layout.status();
    }

    auto effective_logger = logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("frame-ipc");
    try {
        if (options.remove_existing) {
            bip::shared_memory_object::remove(options.name.c_str());
        }
        bip::shared_memory_object object(bip::create_only, options.name.c_str(), bip::read_write);
        object.truncate(static_cast<bip::offset_t>(layout.value().region_size));
        bip::mapped_region region(object, bip::read_write);
        std::memset(region.get_address(), 0, region.get_size());

        auto* header = std::construct_at(static_cast<RegionHeader*>(region.get_address()));
        header->header_size = sizeof(RegionHeader);
        header->region_size = layout.value().region_size;
        header->slot_headers_offset = layout.value().slot_headers_offset;
        header->payload_offset = layout.value().payload_offset;
        header->slot_count = static_cast<std::uint32_t>(options.slot_count);
        header->payload_capacity = static_cast<std::uint32_t>(options.payload_capacity);
        header->epoch.store(GenerateEpoch(), std::memory_order_relaxed);

        auto* slots = reinterpret_cast<SlotHeader*>(
            static_cast<std::byte*>(region.get_address()) + layout.value().slot_headers_offset);
        for (std::size_t index = 0; index < options.slot_count; ++index) {
            auto* slot = std::construct_at(&slots[index]);
            slot->sequence.store(index, std::memory_order_relaxed);
        }
        header->state.store(kRegionReady, std::memory_order_release);

        auto impl = std::make_shared<Impl>(
            std::move(options),
            std::move(object),
            std::move(region),
            effective_logger);
        return std::unique_ptr<SharedMemoryInferenceFrameChannel>(
            new SharedMemoryInferenceFrameChannel(std::move(impl)));
    } catch (const bip::interprocess_exception& exception) {
        return InterprocessFailure(effective_logger, "create shared frame channel", exception);
    }
}

core::Result<std::unique_ptr<SharedMemoryInferenceFrameChannel>>
SharedMemoryInferenceFrameChannel::Open(
    InferenceFrameSharedMemoryOptions options,
    core::LoggerAdapter logger) {
    if (options.name.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "shared frame channel name is required");
    }

    auto effective_logger = logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("frame-ipc");
    try {
        bip::shared_memory_object object(bip::open_only, options.name.c_str(), bip::read_write);
        bip::mapped_region region(object, bip::read_write);
        if (region.get_size() < sizeof(RegionHeader)) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "shared frame region is too small");
        }

        auto* header = static_cast<RegionHeader*>(region.get_address());
        if (header->state.load(std::memory_order_acquire) != kRegionReady ||
            header->magic != kRegionMagic ||
            header->version != kRegionVersion ||
            header->header_size != sizeof(RegionHeader)) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "shared frame region header is incompatible");
        }
        auto layout = CalculateLayout(header->slot_count, header->payload_capacity);
        if (!layout.ok() ||
            header->region_size != layout.value().region_size ||
            header->slot_headers_offset != layout.value().slot_headers_offset ||
            header->payload_offset != layout.value().payload_offset ||
            region.get_size() < header->region_size) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "shared frame region layout is incompatible");
        }

        options.slot_count = header->slot_count;
        options.payload_capacity = header->payload_capacity;
        auto impl = std::make_shared<Impl>(
            std::move(options),
            std::move(object),
            std::move(region),
            effective_logger);
        return std::unique_ptr<SharedMemoryInferenceFrameChannel>(
            new SharedMemoryInferenceFrameChannel(std::move(impl)));
    } catch (const bip::interprocess_exception& exception) {
        return InterprocessFailure(effective_logger, "open shared frame channel", exception);
    }
}

core::Status SharedMemoryInferenceFrameChannel::Remove(std::string_view name) {
    if (name.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "shared frame channel name is required");
    }
    try {
        bip::shared_memory_object::remove(std::string(name).c_str());
        return core::Status::Ok();
    } catch (const bip::interprocess_exception&) {
        return core::Status::Error(core::ErrorCode::Unavailable, "remove shared frame channel failed");
    }
}

SharedMemoryInferenceFrameChannel::SharedMemoryInferenceFrameChannel(std::shared_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

SharedMemoryInferenceFrameChannel::~SharedMemoryInferenceFrameChannel() = default;

core::Status SharedMemoryInferenceFrameChannel::Publish(const SharedFramePublishRequest& request) {
    return impl_->Publish(request);
}

core::Result<ClaimedSharedFrame> SharedMemoryInferenceFrameChannel::TryClaim() {
    return impl_->TryClaim();
}

void SharedMemoryInferenceFrameChannel::Shutdown() {
    impl_->Shutdown();
}

InferenceFrameSharedMemorySnapshot SharedMemoryInferenceFrameChannel::Snapshot() const {
    return impl_->Snapshot();
}

} // namespace ipc::media
