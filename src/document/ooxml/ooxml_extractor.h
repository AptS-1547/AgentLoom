#pragma once

#include "document_types.h"
#include "result.h"
#include "unique_handle.h"

#include <filesystem>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

struct zip;
using zip_t = struct zip;

namespace core {

struct ZipArchiveResource;

template <>
struct ResourceTraits<ZipArchiveResource> {
    using handle_type = zip_t*;
    static handle_type invalid() noexcept { return nullptr; }
    static bool valid(handle_type handle) noexcept { return handle != nullptr; }
    static void close(handle_type handle) noexcept;
};

} // namespace core

namespace agent::document {

struct ZipEntryInfo {
    std::string name;
    std::uint64_t size = 0;
    std::uint64_t compressed_size = 0;
};

class ZipArchive {
public:
    static core::Result<ZipArchive> Open(const std::filesystem::path& path, DocumentExtractOptions options = {});

    ZipArchive() = default;
    ZipArchive(ZipArchive&&) noexcept = default;
    ZipArchive& operator=(ZipArchive&&) noexcept = default;
    ZipArchive(const ZipArchive&) = delete;
    ZipArchive& operator=(const ZipArchive&) = delete;

    core::Result<std::vector<std::string>> ListEntries() const;
    core::Result<std::vector<ZipEntryInfo>> ListEntryInfo() const;
    core::Result<std::string> ReadTextEntry(std::string_view name) const;
    core::Result<std::string> ReadEntry(std::string_view name, std::size_t max_bytes) const;
    core::Status VerifyReadable() const;

private:
    ZipArchive(core::UniqueHandle<core::ZipArchiveResource> handle,
               DocumentExtractOptions options,
               std::shared_ptr<std::string> archive_bytes = {});

    core::UniqueHandle<core::ZipArchiveResource> handle_;
    DocumentExtractOptions options_;
    std::shared_ptr<std::string> archive_bytes_;
};

class OoxmlExtractor {
public:
    explicit OoxmlExtractor(DocumentExtractOptions options = {});

    core::Result<std::vector<DocumentBlock>> ExtractDocxBlocks(const std::filesystem::path& path) const;
    core::Result<std::vector<DocumentBlock>> ExtractPptxBlocks(const std::filesystem::path& path) const;

private:
    DocumentExtractOptions options_;
};

} // namespace agent::document
