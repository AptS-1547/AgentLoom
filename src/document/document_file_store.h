#pragma once

#include "document_metadata_repository.h"
#include "result.h"

#include <chrono>
#include <filesystem>
#include <string>
#include <string_view>

namespace agent::document {

struct DocumentFileStoreOptions {
    std::filesystem::path root_dir;
    std::chrono::hours retention = std::chrono::hours(24 * 7);
};

class DocumentFileStore {
public:
    explicit DocumentFileStore(DocumentFileStoreOptions options);

    core::Status EnsureRoot() const;
    core::Result<std::filesystem::path> ResolveManagedPath(std::string_view stored_path) const;
    core::Result<DocumentMetadataRecord> ImportLocalFile(const std::filesystem::path& source_path,
                                                         std::string_view display_name,
                                                         std::string_view owner_user_uuid = {},
                                                         std::string_view session_id = {}) const;
    core::Status RemoveManagedFile(std::string_view stored_path) const;

    const std::filesystem::path& root_dir() const noexcept { return root_dir_; }
    std::chrono::hours retention() const noexcept { return retention_; }

private:
    std::filesystem::path root_dir_;
    std::chrono::hours retention_;
};

} // namespace agent::document
