#include "inference_frame_spool.h"

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <utility>
#include <unordered_set>
#include <vector>

namespace media::inference {
namespace {

namespace bip = boost::interprocess;

constexpr std::uint64_t kSegmentMagic = 0x414C53504F4F4C31ULL;
constexpr std::uint32_t kRecordMagic = 0x46525031U;
constexpr std::uint32_t kSchemaVersion = 2;
constexpr std::size_t kRecordAlignment = 8;
constexpr std::size_t kMaxIdentityBytes = 4096;

struct alignas(64) SegmentHeader {
    std::uint64_t magic = kSegmentMagic;
    std::uint32_t version = kSchemaVersion;
    std::uint32_t header_size = sizeof(SegmentHeader);
    std::uint64_t segment_index = 0;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t used_bytes = sizeof(SegmentHeader);
    std::uint64_t record_count = 0;
    std::uint32_t sealed = 0;
    std::uint32_t reserved = 0;
};

struct RecordHeader {
    std::uint32_t magic = kRecordMagic;
    std::uint32_t header_size = sizeof(RecordHeader);
    std::uint64_t total_size = 0;
    std::uint64_t selected_sequence = 0;
    std::uint64_t transport_sequence = 0;
    std::uint64_t frame_id = 0;
    std::int64_t timestamp_us = 0;
    std::int64_t published_at_unix_us = 0;
    std::int64_t received_at_unix_us = 0;
    std::int64_t admitted_at_unix_us = 0;
    std::int64_t spooled_at_unix_us = 0;
    std::int64_t replayed_at_unix_us = 0;
    std::int64_t inference_started_at_unix_us = 0;
    std::int64_t terminal_at_unix_us = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format = 0;
    std::uint32_t session_id_size = 0;
    std::uint32_t trace_id_size = 0;
    std::uint32_t payload_size = 0;
    double saliency = 0.0;
    std::uint64_t payload_checksum = 0;
};

static_assert(std::is_trivially_copyable_v<SegmentHeader>);
static_assert(std::is_trivially_copyable_v<RecordHeader>);

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

std::uint64_t Fnv1a(std::span<const std::byte> bytes) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto value : bytes) {
        hash ^= static_cast<std::uint8_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t HashIdentity(std::string_view value) noexcept {
    return Fnv1a(std::as_bytes(std::span(value.data(), value.size())));
}

std::string ExecutionDirectoryName(std::string_view execution_id) {
    std::ostringstream out;
    out << "execution-" << std::hex << std::setw(16) << std::setfill('0') << HashIdentity(execution_id)
        << '-' << std::dec << execution_id.size();
    return out.str();
}

std::string SegmentFileName(std::size_t index) {
    std::ostringstream out;
    out << "segment-" << std::setw(6) << std::setfill('0') << index << ".bin";
    return out.str();
}

std::string PathUtf8(const std::filesystem::path& path) {
    const auto value = path.u8string();
    return std::string(reinterpret_cast<const char*>(value.data()), value.size());
}

core::Status ResizeFile(const std::filesystem::path& path, std::size_t size) {
    if (size == 0 || size > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max())) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "spool segment size is invalid");
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        return core::Status::Error(core::ErrorCode::Unavailable, "failed to create spool segment");
    }
    file.seekp(static_cast<std::streamoff>(size - 1));
    file.put('\0');
    file.flush();
    if (!file) {
        return core::Status::Error(core::ErrorCode::Unavailable, "failed to resize spool segment");
    }
    return core::Status::Ok();
}

core::Status ValidateOptions(const MappedInferenceFrameSpoolOptions& options) {
    if (options.root_directory.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "spool root directory is required");
    }
    if (options.execution_id.empty() || options.execution_id.size() > kMaxIdentityBytes) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "spool execution_id is invalid");
    }
    if (options.segment_bytes <= sizeof(SegmentHeader) + sizeof(RecordHeader)) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "spool segment size is too small");
    }
    if (options.max_spool_bytes < options.segment_bytes) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "max spool bytes must fit one segment");
    }
    return core::Status::Ok();
}

struct SegmentMapping {
    std::filesystem::path path;
    std::size_t index = 0;
    bip::file_mapping mapping;
    bip::mapped_region region;

    SegmentMapping(std::filesystem::path segment_path,
                   std::size_t segment_index,
                   bip::file_mapping file_mapping,
                   bip::mapped_region mapped_region)
        : path(std::move(segment_path)),
          index(segment_index),
          mapping(std::move(file_mapping)),
          region(std::move(mapped_region)) {}

    std::byte* data() noexcept {
        return static_cast<std::byte*>(region.get_address());
    }

    const std::byte* data() const noexcept {
        return static_cast<const std::byte*>(region.get_address());
    }

    SegmentHeader* header() noexcept {
        return reinterpret_cast<SegmentHeader*>(data());
    }

    const SegmentHeader* header() const noexcept {
        return reinterpret_cast<const SegmentHeader*>(data());
    }
};

core::Result<std::shared_ptr<SegmentMapping>> CreateSegment(
    const std::filesystem::path& directory,
    std::size_t index,
    std::size_t segment_bytes) {
    const auto path = directory / SegmentFileName(index);
    auto resize = ResizeFile(path, segment_bytes);
    if (!resize.ok()) {
        return resize;
    }
    try {
#ifdef BOOST_INTERPROCESS_WCHAR_NAMED_RESOURCES
        bip::file_mapping mapping(path.c_str(), bip::read_write);
#else
        bip::file_mapping mapping(path.string().c_str(), bip::read_write);
#endif
        bip::mapped_region region(mapping, bip::read_write);
        if (region.get_size() != segment_bytes) {
            return core::Status::Error(core::ErrorCode::InternalError, "mapped spool segment size mismatch");
        }
        auto segment = std::make_shared<SegmentMapping>(
            path,
            index,
            std::move(mapping),
            std::move(region));
        std::memset(segment->data(), 0, sizeof(SegmentHeader));
        SegmentHeader header;
        header.segment_index = index;
        header.capacity_bytes = segment_bytes;
        std::memcpy(segment->data(), &header, sizeof(header));
        if (!segment->region.flush(0, sizeof(header), false)) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to flush spool segment header");
        }
        return segment;
    } catch (const bip::interprocess_exception&) {
        return core::Status::Error(core::ErrorCode::Unavailable, "failed to map spool segment");
    } catch (const std::exception&) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to initialize spool segment");
    }
}

core::Status ValidateRecordMetadata(const MappedInferenceFrameSpoolOptions& options,
                                    const SpoolFrameMetadata& metadata,
                                    std::span<const std::byte> payload) {
    if (metadata.execution_id != options.execution_id) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "spool execution_id does not match");
    }
    if (metadata.selected_sequence == 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "selected_sequence must be positive");
    }
    if (metadata.frame.session_id.empty() ||
        metadata.frame.session_id.size() > kMaxIdentityBytes ||
        metadata.frame.trace_id.size() > kMaxIdentityBytes) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "spool frame identity is invalid");
    }
    if (payload.empty() || payload.size() > std::numeric_limits<std::uint32_t>::max()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "spool frame payload is invalid");
    }
    return core::Status::Ok();
}

} // namespace

class MappedSpoolFrameLease::Impl {
public:
    Impl(std::shared_ptr<SegmentMapping> segment,
         SpoolFrameMetadata metadata,
         const std::byte* payload,
         std::size_t payload_size)
        : segment_(std::move(segment)),
          metadata_(std::move(metadata)),
          payload_(payload),
          payload_size_(payload_size) {}

    std::shared_ptr<SegmentMapping> segment_;
    SpoolFrameMetadata metadata_;
    const std::byte* payload_ = nullptr;
    std::size_t payload_size_ = 0;
};

MappedSpoolFrameLease::MappedSpoolFrameLease() = default;
MappedSpoolFrameLease::~MappedSpoolFrameLease() = default;
MappedSpoolFrameLease::MappedSpoolFrameLease(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
MappedSpoolFrameLease::MappedSpoolFrameLease(MappedSpoolFrameLease&&) noexcept = default;
MappedSpoolFrameLease& MappedSpoolFrameLease::operator=(MappedSpoolFrameLease&&) noexcept = default;

const SpoolFrameMetadata& MappedSpoolFrameLease::metadata() const noexcept {
    static const SpoolFrameMetadata empty;
    return impl_ ? impl_->metadata_ : empty;
}

std::span<const std::byte> MappedSpoolFrameLease::bytes() const noexcept {
    if (!impl_ || !impl_->payload_) {
        return {};
    }
    return {impl_->payload_, impl_->payload_size_};
}

bool MappedSpoolFrameLease::valid() const noexcept {
    return impl_ && impl_->payload_ && impl_->payload_size_ > 0;
}

class MappedInferenceFrameSpool::Impl {
public:
    Impl(MappedInferenceFrameSpoolOptions options,
         std::filesystem::path directory,
         core::LoggerAdapter logger)
        : options_(std::move(options)),
          directory_(std::move(directory)),
          logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("media-spool")) {}

    ~Impl() {
        if (options_.remove_on_destroy && !cleaned_) {
            const auto status = Cleanup();
            if (!status.ok()) {
                logger_.warn("[media-spool] cleanup deferred directory={} code={} message={}",
                             PathUtf8(directory_),
                             static_cast<int>(status.code()),
                             status.message());
            }
        }
    }

    core::Status Initialize() {
        std::error_code ec;
        std::filesystem::create_directories(options_.root_directory, ec);
        if (ec) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to create spool root directory");
        }
        if (std::filesystem::exists(directory_, ec)) {
            return core::Status::Error(core::ErrorCode::AlreadyExists, "spool execution directory already exists");
        }
        if (!std::filesystem::create_directory(directory_, ec) || ec) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to create spool execution directory");
        }
        return core::Status::Ok();
    }

    core::Result<SpoolRecordHandle> Append(
        const SpoolFrameMetadata& metadata,
        std::span<const std::byte> payload) {
        auto validation = ValidateRecordMetadata(options_, metadata, payload);
        if (!validation.ok()) {
            return validation;
        }

        std::lock_guard lock(mutex_);
        if (cleaned_) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "spool is cleaned");
        }
        if (sealed_) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "spool is sealed");
        }
        if (selected_sequences_.contains(metadata.selected_sequence)) {
            return core::Status::Error(core::ErrorCode::AlreadyExists, "selected_sequence already exists");
        }

        const auto variable_bytes = metadata.frame.session_id.size() + metadata.frame.trace_id.size() + payload.size();
        if (variable_bytes > std::numeric_limits<std::size_t>::max() - sizeof(RecordHeader)) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "spool record size overflow");
        }
        const auto record_bytes = AlignUp(sizeof(RecordHeader) + variable_bytes, kRecordAlignment);
        if (record_bytes > options_.segment_bytes - sizeof(SegmentHeader)) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "spool record exceeds segment capacity");
        }

        if (segments_.empty()) {
            auto add = AddSegmentLocked();
            if (!add.ok()) {
                return add;
            }
        }
        auto segment = segments_.back();
        auto* segment_header = segment->header();
        if (segment_header->used_bytes + record_bytes > segment_header->capacity_bytes) {
            auto seal = SealSegmentLocked(*segment);
            if (!seal.ok()) {
                return seal;
            }
            auto add = AddSegmentLocked();
            if (!add.ok()) {
                return add;
            }
            segment = segments_.back();
            segment_header = segment->header();
        }

        const auto record_offset = static_cast<std::size_t>(segment_header->used_bytes);
        RecordHeader record;
        record.total_size = record_bytes;
        record.selected_sequence = metadata.selected_sequence;
        record.transport_sequence = metadata.frame.transport_sequence;
        record.frame_id = metadata.frame.frame_id;
        record.timestamp_us = metadata.frame.timestamp_us;
        record.published_at_unix_us = metadata.frame.timing.published_at_unix_us;
        record.received_at_unix_us = metadata.frame.timing.received_at_unix_us;
        record.admitted_at_unix_us = metadata.frame.timing.admitted_at_unix_us;
        record.spooled_at_unix_us = metadata.frame.timing.spooled_at_unix_us;
        record.replayed_at_unix_us = metadata.frame.timing.replayed_at_unix_us;
        record.inference_started_at_unix_us = metadata.frame.timing.inference_started_at_unix_us;
        record.terminal_at_unix_us = metadata.frame.timing.terminal_at_unix_us;
        record.width = metadata.frame.width;
        record.height = metadata.frame.height;
        record.format = static_cast<std::uint32_t>(metadata.frame.format);
        record.session_id_size = static_cast<std::uint32_t>(metadata.frame.session_id.size());
        record.trace_id_size = static_cast<std::uint32_t>(metadata.frame.trace_id.size());
        record.payload_size = static_cast<std::uint32_t>(payload.size());
        record.saliency = metadata.frame.saliency;
        record.payload_checksum = Fnv1a(payload);

        selected_sequences_.insert(metadata.selected_sequence);
        auto* destination = segment->data() + record_offset;
        std::memset(destination, 0, record_bytes);
        std::memcpy(destination, &record, sizeof(record));
        auto* cursor = destination + sizeof(record);
        std::memcpy(cursor, metadata.frame.session_id.data(), metadata.frame.session_id.size());
        cursor += metadata.frame.session_id.size();
        if (!metadata.frame.trace_id.empty()) {
            std::memcpy(cursor, metadata.frame.trace_id.data(), metadata.frame.trace_id.size());
        }
        cursor += metadata.frame.trace_id.size();
        std::memcpy(cursor, payload.data(), payload.size());

        if (options_.flush_on_append && !segment->region.flush(record_offset, record_bytes, false)) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to flush spool record");
        }
        segment_header->used_bytes += record_bytes;
        ++segment_header->record_count;
        if (options_.flush_on_append && !segment->region.flush(0, sizeof(SegmentHeader), false)) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to flush spool segment metadata");
        }

        ++admitted_records_;
        payload_bytes_ += payload.size();
        return SpoolRecordHandle{
            .selected_sequence = metadata.selected_sequence,
            .segment_index = segment->index,
            .record_offset = record_offset,
            .payload_size = payload.size(),
        };
    }

    core::Status Seal() {
        std::lock_guard lock(mutex_);
        if (cleaned_) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "spool is cleaned");
        }
        if (sealed_) {
            return core::Status::Ok();
        }
        if (!segments_.empty()) {
            auto status = SealSegmentLocked(*segments_.back());
            if (!status.ok()) {
                return status;
            }
        }
        sealed_ = true;
        return core::Status::Ok();
    }

    core::Result<std::optional<MappedSpoolFrameLease>> ReplayNext() {
        std::lock_guard lock(mutex_);
        if (cleaned_) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "spool is cleaned");
        }
        if (!sealed_) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "spool must be sealed before replay");
        }

        while (replay_segment_index_ < segments_.size()) {
            auto segment = segments_[replay_segment_index_];
            const auto* segment_header = segment->header();
            if (segment_header->magic != kSegmentMagic ||
                segment_header->version != kSchemaVersion ||
                segment_header->header_size != sizeof(SegmentHeader) ||
                segment_header->used_bytes > segment_header->capacity_bytes ||
                segment_header->sealed == 0) {
                return core::Status::Error(core::ErrorCode::InternalError, "spool segment header is corrupt");
            }
            if (replay_offset_ >= segment_header->used_bytes) {
                ++replay_segment_index_;
                replay_offset_ = sizeof(SegmentHeader);
                continue;
            }
            if (segment_header->used_bytes - replay_offset_ < sizeof(RecordHeader)) {
                return core::Status::Error(core::ErrorCode::InternalError, "spool record header is truncated");
            }

            RecordHeader record;
            std::memcpy(&record, segment->data() + replay_offset_, sizeof(record));
            const auto minimum_size = sizeof(RecordHeader) +
                                      static_cast<std::size_t>(record.session_id_size) +
                                      static_cast<std::size_t>(record.trace_id_size) +
                                      static_cast<std::size_t>(record.payload_size);
            if (record.magic != kRecordMagic ||
                record.header_size != sizeof(RecordHeader) ||
                record.total_size < minimum_size ||
                record.total_size % kRecordAlignment != 0 ||
                record.total_size > segment_header->used_bytes - replay_offset_) {
                return core::Status::Error(core::ErrorCode::InternalError, "spool record is corrupt");
            }

            const auto* cursor = segment->data() + replay_offset_ + sizeof(RecordHeader);
            SpoolFrameMetadata metadata;
            metadata.execution_id = options_.execution_id;
            metadata.selected_sequence = record.selected_sequence;
            metadata.frame.execution_id = options_.execution_id;
            metadata.frame.selected_sequence = record.selected_sequence;
            metadata.frame.transport_sequence = record.transport_sequence;
            metadata.frame.session_id.assign(
                reinterpret_cast<const char*>(cursor),
                record.session_id_size);
            cursor += record.session_id_size;
            metadata.frame.trace_id.assign(
                reinterpret_cast<const char*>(cursor),
                record.trace_id_size);
            cursor += record.trace_id_size;
            metadata.frame.frame_id = record.frame_id;
            metadata.frame.timestamp_us = record.timestamp_us;
            metadata.frame.timing.published_at_unix_us = record.published_at_unix_us;
            metadata.frame.timing.received_at_unix_us = record.received_at_unix_us;
            metadata.frame.timing.admitted_at_unix_us = record.admitted_at_unix_us;
            metadata.frame.timing.spooled_at_unix_us = record.spooled_at_unix_us;
            metadata.frame.timing.replayed_at_unix_us = record.replayed_at_unix_us;
            metadata.frame.timing.inference_started_at_unix_us = record.inference_started_at_unix_us;
            metadata.frame.timing.terminal_at_unix_us = record.terminal_at_unix_us;
            metadata.frame.width = record.width;
            metadata.frame.height = record.height;
            metadata.frame.format = static_cast<InferenceFrameFormat>(record.format);
            metadata.frame.saliency = record.saliency;

            const auto payload = std::span<const std::byte>(cursor, record.payload_size);
            if (Fnv1a(payload) != record.payload_checksum) {
                return core::Status::Error(core::ErrorCode::InternalError, "spool payload checksum mismatch");
            }

            auto lease = MappedSpoolFrameLease(std::make_unique<MappedSpoolFrameLease::Impl>(
                std::move(segment),
                std::move(metadata),
                payload.data(),
                payload.size()));
            replay_offset_ += static_cast<std::size_t>(record.total_size);
            ++replayed_records_;
            return std::optional<MappedSpoolFrameLease>(std::move(lease));
        }
        return std::optional<MappedSpoolFrameLease>{};
    }

    MappedInferenceFrameSpoolSnapshot Snapshot() const {
        std::lock_guard lock(mutex_);
        return {
            .segment_count = segments_.size(),
            .admitted_records = admitted_records_,
            .replayed_records = replayed_records_,
            .payload_bytes = payload_bytes_,
            .allocated_bytes = allocated_bytes_,
            .sealed = sealed_,
            .cleaned = cleaned_,
        };
    }

    core::Status Cleanup() {
        std::lock_guard lock(mutex_);
        if (cleaned_) {
            return core::Status::Ok();
        }
        for (const auto& segment : segments_) {
            if (segment.use_count() > 1) {
                return core::Status::Error(core::ErrorCode::FailedPrecondition, "spool read leases are still active");
            }
        }
        segments_.clear();
        std::error_code ec;
        std::filesystem::remove_all(directory_, ec);
        if (ec) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to remove spool execution directory");
        }
        cleaned_ = true;
        return core::Status::Ok();
    }

private:
    core::Status AddSegmentLocked() {
        if (allocated_bytes_ > options_.max_spool_bytes - options_.segment_bytes) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "spool byte limit reached");
        }
        auto segment = CreateSegment(directory_, segments_.size(), options_.segment_bytes);
        if (!segment.ok()) {
            return segment.status();
        }
        segments_.push_back(std::move(segment).value());
        allocated_bytes_ += options_.segment_bytes;
        return core::Status::Ok();
    }

    core::Status SealSegmentLocked(SegmentMapping& segment) {
        auto* header = segment.header();
        if (header->sealed != 0) {
            return core::Status::Ok();
        }
        header->sealed = 1;
        if (!segment.region.flush(0, sizeof(SegmentHeader), false)) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to seal spool segment");
        }
        return core::Status::Ok();
    }

    MappedInferenceFrameSpoolOptions options_;
    std::filesystem::path directory_;
    core::LoggerAdapter logger_;
    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<SegmentMapping>> segments_;
    std::size_t admitted_records_ = 0;
    std::size_t replayed_records_ = 0;
    std::size_t payload_bytes_ = 0;
    std::size_t allocated_bytes_ = 0;
    std::unordered_set<std::uint64_t> selected_sequences_;
    std::size_t replay_segment_index_ = 0;
    std::size_t replay_offset_ = sizeof(SegmentHeader);
    bool sealed_ = false;
    bool cleaned_ = false;
};

core::Result<std::unique_ptr<MappedInferenceFrameSpool>> MappedInferenceFrameSpool::Create(
    MappedInferenceFrameSpoolOptions options,
    core::LoggerAdapter logger) {
    auto validation = ValidateOptions(options);
    if (!validation.ok()) {
        return validation;
    }
    const auto directory = options.root_directory / ExecutionDirectoryName(options.execution_id);
    try {
        auto impl = std::make_unique<Impl>(std::move(options), directory, std::move(logger));
        auto initialize = impl->Initialize();
        if (!initialize.ok()) {
            return initialize;
        }
        return std::unique_ptr<MappedInferenceFrameSpool>(
            new MappedInferenceFrameSpool(std::move(impl)));
    } catch (const std::bad_alloc&) {
        return core::Status::Error(core::ErrorCode::OutOfMemory, "failed to allocate mapped spool");
    } catch (const std::exception&) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to create mapped spool");
    }
}

MappedInferenceFrameSpool::MappedInferenceFrameSpool(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
MappedInferenceFrameSpool::~MappedInferenceFrameSpool() = default;

core::Result<SpoolRecordHandle> MappedInferenceFrameSpool::Append(
    const SpoolFrameMetadata& metadata,
    std::span<const std::byte> encoded_payload) {
    try {
        return impl_->Append(metadata, encoded_payload);
    } catch (const std::bad_alloc&) {
        return core::Status::Error(core::ErrorCode::OutOfMemory, "failed to append mapped spool record");
    } catch (const std::exception&) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to append mapped spool record");
    }
}

core::Status MappedInferenceFrameSpool::Seal() {
    return impl_->Seal();
}

core::Result<std::optional<MappedSpoolFrameLease>> MappedInferenceFrameSpool::ReplayNext() {
    try {
        return impl_->ReplayNext();
    } catch (const std::bad_alloc&) {
        return core::Status::Error(core::ErrorCode::OutOfMemory, "failed to replay mapped spool record");
    } catch (const std::exception&) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to replay mapped spool record");
    }
}

MappedInferenceFrameSpoolSnapshot MappedInferenceFrameSpool::Snapshot() const {
    return impl_->Snapshot();
}

core::Status MappedInferenceFrameSpool::Cleanup() {
    return impl_->Cleanup();
}

} // namespace media::inference
