#include "document_file_store.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>

namespace agent::document {
namespace {

std::int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string PathToUtf8(const std::filesystem::path& path) {
    const auto utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

std::filesystem::path PathFromUtf8(std::string_view path) {
#ifdef _WIN32
    std::u8string utf8;
    utf8.reserve(path.size());
    for (char ch : path) {
        utf8.push_back(static_cast<char8_t>(ch));
    }
    return std::filesystem::path(std::move(utf8));
#else
    return std::filesystem::path(std::string(path));
#endif
}

std::string LowerExtension(const std::filesystem::path& path) {
    auto ext_u8 = path.extension().u8string();
    std::string ext(ext_u8.begin(), ext_u8.end());
    if (!ext.empty() && ext.front() == '.') {
        ext.erase(ext.begin());
    }
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext;
}

std::string DisplayNameOrFileName(const std::filesystem::path& path, std::string_view display_name) {
    if (!display_name.empty()) {
        return std::string(display_name);
    }
    return PathToUtf8(path.filename());
}

std::string Hex64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

core::Result<std::string> FileContentHash(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return core::Status::Error(core::ErrorCode::NotFound, "document source file cannot be opened");
    }

    std::uint64_t hash = 1469598103934665603ull;
    std::array<char, 64 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto read = input.gcount();
        for (std::streamsize i = 0; i < read; ++i) {
            hash ^= static_cast<unsigned char>(buffer[static_cast<std::size_t>(i)]);
            hash *= 1099511628211ull;
        }
    }
    if (input.bad()) {
        return core::Status::Error(core::ErrorCode::InternalError, "document source file read failed");
    }
    return Hex64(hash);
}

bool IsInsideOrEqual(const std::filesystem::path& child, const std::filesystem::path& parent) {
    auto child_it = child.begin();
    auto parent_it = parent.begin();
    for (; parent_it != parent.end(); ++parent_it, ++child_it) {
        if (child_it == child.end() || *child_it != *parent_it) {
            return false;
        }
    }
    return true;
}

} // namespace

DocumentFileStore::DocumentFileStore(DocumentFileStoreOptions options)
    : root_dir_(std::move(options.root_dir)),
      retention_(options.retention) {}

core::Status DocumentFileStore::EnsureRoot() const {
    if (root_dir_.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "document file store root is required");
    }
    std::error_code ec;
    std::filesystem::create_directories(root_dir_, ec);
    if (ec) {
        return core::Status::Error(core::ErrorCode::InternalError, "create document file store root failed: " + ec.message());
    }
    return core::Status::Ok();
}

core::Result<std::filesystem::path> DocumentFileStore::ResolveManagedPath(std::string_view stored_path) const {
    if (stored_path.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "managed document path is required");
    }
    std::error_code ec;
    const auto root = std::filesystem::weakly_canonical(root_dir_, ec);
    if (ec) {
        return core::Status::Error(core::ErrorCode::InternalError, "canonical document root failed: " + ec.message());
    }
    auto candidate = PathFromUtf8(stored_path);
    if (candidate.is_relative()) {
        candidate = root / candidate;
    }
    const auto resolved = std::filesystem::weakly_canonical(candidate, ec);
    if (ec) {
        return core::Status::Error(core::ErrorCode::NotFound, "resolve managed document path failed: " + ec.message());
    }
    if (!IsInsideOrEqual(resolved, root)) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "managed document path escapes document root");
    }
    return resolved;
}

core::Result<DocumentMetadataRecord> DocumentFileStore::ImportLocalFile(const std::filesystem::path& source_path,
                                                                        std::string_view display_name,
                                                                        std::string_view owner_user_uuid,
                                                                        std::string_view session_id) const {
    if (auto status = EnsureRoot(); !status.ok()) {
        return status;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(source_path, ec)) {
        return core::Status::Error(core::ErrorCode::NotFound, "document source file is not a regular file");
    }
    auto hash = FileContentHash(source_path);
    if (!hash.ok()) {
        return hash.status();
    }

    DocumentMetadataRecord record;
    record.content_hash = std::move(hash).value();
    record.document_id = "doc_" + record.content_hash;
    record.file_name = DisplayNameOrFileName(source_path, display_name);
    auto display_path = PathFromUtf8(record.file_name);
    record.file_type = LowerExtension(display_path);
    if (record.file_type.empty()) {
        record.file_type = LowerExtension(source_path);
    }
    record.owner_user_uuid = std::string(owner_user_uuid);
    record.session_id = std::string(session_id);
    record.uploaded_at_ms = NowMs();
    record.last_accessed_at_ms = record.uploaded_at_ms;
    record.analysis_status = "uploaded";
    record.size_bytes = static_cast<std::int64_t>(std::filesystem::file_size(source_path, ec));
    if (ec) {
        return core::Status::Error(core::ErrorCode::InternalError, "read document file size failed: " + ec.message());
    }

    std::filesystem::path stored_relative = record.document_id;
    if (!record.file_type.empty()) {
        stored_relative += "." + record.file_type;
    }
    const auto target = root_dir_ / stored_relative;
    std::filesystem::copy_file(source_path, target, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
        return core::Status::Error(core::ErrorCode::InternalError, "copy document into managed store failed: " + ec.message());
    }
    record.storage_path = PathToUtf8(stored_relative);
    return record;
}

core::Status DocumentFileStore::RemoveManagedFile(std::string_view stored_path) const {
    auto resolved = ResolveManagedPath(stored_path);
    if (!resolved.ok()) {
        return resolved.status();
    }
    std::error_code ec;
    std::filesystem::remove(resolved.value(), ec);
    if (ec) {
        return core::Status::Error(core::ErrorCode::InternalError, "remove managed document failed: " + ec.message());
    }
    return core::Status::Ok();
}

} // namespace agent::document
