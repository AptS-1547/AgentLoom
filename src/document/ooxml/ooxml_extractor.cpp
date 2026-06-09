#include "ooxml_extractor.h"

#include <pugixml.hpp>
#include <zip.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>

namespace core {

void ResourceTraits<ZipArchiveResource>::close(handle_type handle) noexcept {
    zip_close(handle);
}

} // namespace core

namespace agent::document {
namespace {

std::string_view LocalName(std::string_view name) {
    const auto pos = name.find(':');
    if (pos != std::string_view::npos) {
        return name.substr(pos + 1);
    }
    if (name.size() >= 3 && name.front() == '{') {
        const auto end = name.find('}');
        if (end != std::string_view::npos) {
            return name.substr(end + 1);
        }
    }
    return name;
}

std::string LowerAscii(std::string_view value) {
    std::string out(value);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

bool StartsWithAny(std::string_view value, std::initializer_list<std::string_view> prefixes) {
    for (const auto prefix : prefixes) {
        if (value.starts_with(prefix)) {
            return true;
        }
    }
    return false;
}

bool EndsWithAny(std::string_view value, std::initializer_list<std::string_view> suffixes) {
    for (const auto suffix : suffixes) {
        if (value.ends_with(suffix)) {
            return true;
        }
    }
    return false;
}

bool HasTag(const pugi::xml_node& node, std::string_view name) {
    return LocalName(node.name()) == name;
}

bool HasAttrLocalName(const pugi::xml_attribute& attr, std::string_view name) {
    return LocalName(attr.name()) == name;
}

std::string AttrValue(const pugi::xml_node& node, std::string_view name) {
    if (auto attr = node.attribute(std::string(name).c_str())) {
        return attr.value();
    }
    for (auto attr : node.attributes()) {
        if (HasAttrLocalName(attr, name)) {
            return attr.value();
        }
    }
    return {};
}

std::optional<int> AttrInt(const pugi::xml_node& node, std::string_view name) {
    return SafeInt(AttrValue(node, name));
}

pugi::xml_node FirstChildByLocalName(const pugi::xml_node& node, std::string_view name) {
    for (auto child : node.children()) {
        if (HasTag(child, name)) {
            return child;
        }
    }
    return {};
}

template <typename Fn>
void ForEachDescendant(const pugi::xml_node& node, Fn&& fn) {
    for (auto child : node.children()) {
        fn(child);
        ForEachDescendant(child, fn);
    }
}

template <typename Fn>
void ForEachDescendantByLocalName(const pugi::xml_node& node, std::string_view name, Fn&& fn) {
    ForEachDescendant(node, [&](const pugi::xml_node& child) {
        if (HasTag(child, name)) {
            fn(child);
        }
    });
}

pugi::xml_document ParseXml(std::string_view xml, core::Status* status) {
    pugi::xml_document doc;
    auto result = doc.load_buffer(xml.data(), xml.size(), pugi::parse_default, pugi::encoding_utf8);
    if (!result) {
        *status = core::Status::Error(core::ErrorCode::InvalidArgument, result.description());
    }
    return doc;
}

std::string DocxParagraphText(const pugi::xml_node& paragraph) {
    std::string out;
    ForEachDescendant(paragraph, [&](const pugi::xml_node& node) {
        if (HasTag(node, "t") && node.first_child()) {
            out.append(node.child_value());
        } else if (HasTag(node, "tab")) {
            out.push_back('\t');
        } else if (HasTag(node, "br") || HasTag(node, "cr")) {
            out.push_back('\n');
        }
    });
    return CleanText(out);
}

DocumentBlock DocxBlockFromParagraph(const pugi::xml_node& paragraph,
                                     std::uint64_t order,
                                     int paragraph_index,
                                     std::string_view xml_part,
                                     std::string text) {
    DocumentBlock block;
    block.id = DocumentBlockId("docx", order);
    block.text = std::move(text);
    block.source = "docx";
    block.order = order;
    block.paragraph_index = paragraph_index;
    block.confidence = 0.8;
    block.metadata["xml_part"] = std::string(xml_part);

    auto ppr = FirstChildByLocalName(paragraph, "pPr");
    if (ppr) {
        auto style_node = FirstChildByLocalName(ppr, "pStyle");
        if (style_node) {
            block.style = AttrValue(style_node, "val");
            block.heading_level = HeadingLevelFromStyle(block.style);
        }

        auto num_pr = FirstChildByLocalName(ppr, "numPr");
        if (num_pr) {
            auto ilvl = FirstChildByLocalName(num_pr, "ilvl");
            if (ilvl) {
                block.bullet_level = SafeInt(AttrValue(ilvl, "val"));
            }
            auto num_id = FirstChildByLocalName(num_pr, "numId");
            if (num_id) {
                block.numbering = AttrValue(num_id, "val");
            }
        }
    }

    ForEachDescendantByLocalName(paragraph, "rPr", [&](const pugi::xml_node& rpr) {
        if (FirstChildByLocalName(rpr, "b")) {
            block.bold = true;
        }
        auto sz = FirstChildByLocalName(rpr, "sz");
        if (sz) {
            auto raw = SafeInt(AttrValue(sz, "val"));
            if (raw && *raw > 0) {
                const auto size = static_cast<double>(*raw) / 2.0;
                block.font_size = block.font_size ? std::max(*block.font_size, size) : size;
            }
        }
    });

    if (auto inferred = InferHeadingLevel(block)) {
        block.heading_level = inferred;
        block.kind = "heading";
        block.confidence = 0.9;
    }
    return block;
}

int PptxSlideNumber(std::string_view name) {
    std::smatch match;
    const auto text = std::string(name);
    if (std::regex_search(text, match, std::regex(R"(slide(\d+)\.xml$)")) && match.size() > 1) {
        return std::stoi(match[1].str());
    }
    return 0;
}

std::string PptxShapeKind(const pugi::xml_node& shape) {
    std::string kind = "paragraph";
    ForEachDescendantByLocalName(shape, "ph", [&](const pugi::xml_node& node) {
        const std::string type = AttrValue(node, "type");
        if (type == "title" || type == "ctrTitle" || type == "subTitle") {
            kind = "heading";
        }
    });
    return kind;
}

std::string PptxParagraphText(const pugi::xml_node& paragraph) {
    std::string out;
    ForEachDescendant(paragraph, [&](const pugi::xml_node& node) {
        if (HasTag(node, "t") && node.first_child()) {
            out.append(node.child_value());
        } else if (HasTag(node, "br")) {
            out.push_back('\n');
        }
    });
    return CleanText(out);
}

std::optional<double> PptxFontSize(const pugi::xml_node& paragraph) {
    std::optional<double> max_size;
    ForEachDescendantByLocalName(paragraph, "rPr", [&](const pugi::xml_node& rpr) {
        if (auto raw = AttrInt(rpr, "sz")) {
            const auto size = static_cast<double>(*raw) / 100.0;
            max_size = max_size ? std::max(*max_size, size) : size;
        }
    });
    return max_size;
}

bool IsDocxExtraXmlPart(std::string_view name) {
    return std::regex_match(
        std::string(name),
        std::regex(R"(word/(header|footer|footnotes|endnotes)\d*\.xml$)"));
}

bool IsPptxSlideXmlPart(std::string_view name) {
    return std::regex_match(std::string(name), std::regex(R"(ppt/slides/slide\d+\.xml$)"));
}

core::Result<std::shared_ptr<std::string>> ReadFileBytes(const std::filesystem::path& path,
                                                         const DocumentExtractOptions& options) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "failed to open document file");
    }
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end < 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "failed to determine document file size");
    }
    const auto size = static_cast<std::uint64_t>(end);
    if (size > options.max_total_uncompressed_bytes) {
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "document file is too large");
    }
    input.seekg(0, std::ios::beg);

    auto bytes = std::make_shared<std::string>();
    bytes->resize(static_cast<std::size_t>(size));
    if (!bytes->empty()) {
        input.read(bytes->data(), static_cast<std::streamsize>(bytes->size()));
        if (input.gcount() != static_cast<std::streamsize>(bytes->size())) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "failed to read complete document file");
        }
    }
    if (bytes->size() >= 2 && (*bytes)[0] == 'M' && (*bytes)[1] == 'Z') {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "self-extracting executable package is not allowed");
    }
    if (bytes->size() < 4
        || static_cast<unsigned char>((*bytes)[0]) != 'P'
        || static_cast<unsigned char>((*bytes)[1]) != 'K'
        || static_cast<unsigned char>((*bytes)[2]) != 0x03
        || static_cast<unsigned char>((*bytes)[3]) != 0x04) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "office document is not a plain ZIP/OOXML package");
    }
    return bytes;
}

core::Status ValidateEntryPolicy(const ZipEntryInfo& entry, const DocumentExtractOptions& options) {
    const auto name = LowerAscii(entry.name);
    if (entry.name.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "zip entry name is empty");
    }
    if (entry.size > options.max_archive_entry_bytes) {
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "zip entry is too large: " + entry.name);
    }
    if (entry.compressed_size > 0 && entry.size > 0) {
        const auto ratio = static_cast<double>(entry.size) / static_cast<double>(entry.compressed_size);
        if (ratio > options.max_compression_ratio) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "zip entry compression ratio is too high: " + entry.name);
        }
    }
    if (options.reject_macros && EndsWithAny(name, {
            "/vbaproject.bin",
            "/vba_project.bin",
            "/vbadata.xml",
            "/vbaprojectsignature.bin"})) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "office macro project is not allowed: " + entry.name);
    }
    if (options.reject_active_content && (
            StartsWithAny(name, {"word/activex/", "xl/activex/", "ppt/activex/"})
            || EndsWithAny(name, {".xlam", ".xla", ".ppam", ".ppa"}))) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "office active content is not allowed: " + entry.name);
    }
    if (options.reject_embedded_packages && (
            StartsWithAny(name, {
                "word/embeddings/",
                "word/oleobjects/",
                "xl/embeddings/",
                "xl/oleobjects/",
                "ppt/embeddings/",
                "ppt/oleobjects/"})
            || EndsWithAny(name, {
                ".exe", ".dll", ".com", ".scr", ".bat", ".cmd", ".ps1", ".vbs",
                ".js", ".jse", ".wsf", ".msi", ".msp", ".hta", ".lnk"}))) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "embedded executable package is not allowed: " + entry.name);
    }
    return core::Status::Ok();
}

core::Status ValidateOoxmlPackage(const ZipArchive& archive,
                                  std::string_view file_type,
                                  const DocumentExtractOptions& options) {
    auto entries = archive.ListEntryInfo();
    if (!entries.ok()) {
        return entries.status();
    }

    bool has_content_types = false;
    bool has_docx_main = false;
    bool has_pptx_slide = false;
    std::uint64_t total_uncompressed = 0;
    for (const auto& entry : entries.value()) {
        total_uncompressed += entry.size;
        if (total_uncompressed > options.max_total_uncompressed_bytes) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "zip archive uncompressed payload is too large");
        }
        if (auto status = ValidateEntryPolicy(entry, options); !status.ok()) {
            return status;
        }
        if (entry.name == "[Content_Types].xml") {
            has_content_types = true;
        }
        if (entry.name == "word/document.xml") {
            has_docx_main = true;
        }
        if (IsPptxSlideXmlPart(entry.name)) {
            has_pptx_slide = true;
        }
    }
    if (!has_content_types) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "OOXML package is missing [Content_Types].xml");
    }
    if (file_type == "docx" && !has_docx_main) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "DOCX package is missing word/document.xml");
    }
    if (file_type == "pptx" && !has_pptx_slide) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "PPTX package has no slide XML parts");
    }
    return archive.VerifyReadable();
}

} // namespace

core::Result<ZipArchive> ZipArchive::Open(const std::filesystem::path& path, DocumentExtractOptions options) {
    auto bytes = ReadFileBytes(path, options);
    if (!bytes.ok()) {
        return bytes.status();
    }

    zip_error_t source_error;
    zip_error_init(&source_error);
    zip_source_t* source = zip_source_buffer_create(
        bytes.value()->data(),
        bytes.value()->size(),
        0,
        &source_error);
    if (!source) {
        std::string message = zip_error_strerror(&source_error);
        zip_error_fini(&source_error);
        return core::Status::Error(core::ErrorCode::InvalidArgument, "failed to create zip memory source: " + message);
    }
    zip_error_fini(&source_error);

    zip_error_t archive_error;
    zip_error_init(&archive_error);
    auto* raw = zip_open_from_source(source, ZIP_RDONLY, &archive_error);
    if (!raw) {
        std::string message = zip_error_strerror(&archive_error);
        zip_error_fini(&archive_error);
        zip_source_free(source);
        return core::Status::Error(core::ErrorCode::InvalidArgument, "failed to open zip archive: " + message);
    }
    zip_error_fini(&archive_error);
    return ZipArchive(core::make_unique_handle<core::ZipArchiveResource>(raw), options, std::move(bytes).value());
}

ZipArchive::ZipArchive(core::UniqueHandle<core::ZipArchiveResource> handle,
                       DocumentExtractOptions options,
                       std::shared_ptr<std::string> archive_bytes)
    : handle_(std::move(handle)),
      options_(options),
      archive_bytes_(std::move(archive_bytes)) {}

core::Result<std::vector<std::string>> ZipArchive::ListEntries() const {
    auto info = ListEntryInfo();
    if (!info.ok()) {
        return info.status();
    }
    std::vector<std::string> names;
    names.reserve(info.value().size());
    for (auto& entry : info.value()) {
        names.push_back(std::move(entry.name));
    }
    return names;
}

core::Result<std::vector<ZipEntryInfo>> ZipArchive::ListEntryInfo() const {
    if (!handle_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "zip archive is not open");
    }
    const auto count = zip_get_num_entries(handle_.get(), ZIP_FL_UNCHANGED);
    if (count < 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, zip_strerror(handle_.get()));
    }
    if (static_cast<std::size_t>(count) > options_.max_zip_entries) {
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "zip archive has too many entries");
    }
    std::vector<ZipEntryInfo> entries;
    entries.reserve(static_cast<std::size_t>(count));
    for (zip_int64_t i = 0; i < count; ++i) {
        const char* name = zip_get_name(handle_.get(), static_cast<zip_uint64_t>(i), ZIP_FL_ENC_GUESS);
        if (!name) {
            continue;
        }
        std::string entry = name;
        if (entry.find("..") != std::string::npos || entry.starts_with('/') || entry.starts_with('\\')) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "unsafe zip entry path");
        }
        zip_stat_t stat{};
        zip_stat_init(&stat);
        if (zip_stat_index(handle_.get(), static_cast<zip_uint64_t>(i), ZIP_FL_UNCHANGED, &stat) != 0) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, zip_strerror(handle_.get()));
        }
        ZipEntryInfo info;
        info.name = std::move(entry);
        info.size = stat.size;
        info.compressed_size = stat.comp_size;
        entries.push_back(std::move(info));
    }
    return entries;
}

core::Result<std::string> ZipArchive::ReadTextEntry(std::string_view name) const {
    return ReadEntry(name, options_.max_xml_entry_bytes);
}

core::Result<std::string> ZipArchive::ReadEntry(std::string_view name, std::size_t max_bytes) const {
    if (!handle_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "zip archive is not open");
    }

    zip_stat_t stat{};
    zip_stat_init(&stat);
    if (zip_stat(handle_.get(), std::string(name).c_str(), ZIP_FL_ENC_GUESS, &stat) != 0) {
        return core::Status::Error(core::ErrorCode::NotFound, "zip entry not found: " + std::string(name));
    }
    if (stat.size > max_bytes) {
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "zip entry is too large: " + std::string(name));
    }

    zip_file_t* file = zip_fopen(handle_.get(), std::string(name).c_str(), ZIP_FL_ENC_GUESS);
    if (!file) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, zip_strerror(handle_.get()));
    }

    std::string out;
    out.resize(static_cast<std::size_t>(stat.size));
    std::size_t offset = 0;
    while (offset < out.size()) {
        const auto read = zip_fread(file, out.data() + offset, out.size() - offset);
        if (read < 0) {
            const auto message = std::string(zip_file_strerror(file));
            zip_fclose(file);
            return core::Status::Error(core::ErrorCode::InvalidArgument, message);
        }
        if (read == 0) {
            break;
        }
        offset += static_cast<std::size_t>(read);
    }
    zip_fclose(file);
    out.resize(offset);
    if (offset != stat.size) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "zip entry was not fully readable: " + std::string(name));
    }
    return out;
}

core::Status ZipArchive::VerifyReadable() const {
    auto entries = ListEntryInfo();
    if (!entries.ok()) {
        return entries.status();
    }
    for (const auto& entry : entries.value()) {
        auto bytes = ReadEntry(entry.name, options_.max_archive_entry_bytes);
        if (!bytes.ok()) {
            return bytes.status();
        }
    }
    return core::Status::Ok();
}

OoxmlExtractor::OoxmlExtractor(DocumentExtractOptions options)
    : options_(options) {}

core::Result<std::vector<DocumentBlock>> OoxmlExtractor::ExtractDocxBlocks(const std::filesystem::path& path) const {
    auto archive = ZipArchive::Open(path, options_);
    if (!archive.ok()) {
        return archive.status();
    }
    if (auto status = ValidateOoxmlPackage(archive.value(), "docx", options_); !status.ok()) {
        return status;
    }
    auto entries = archive.value().ListEntries();
    if (!entries.ok()) {
        return entries.status();
    }

    std::vector<std::string> xml_parts{"word/document.xml"};
    for (const auto& entry : entries.value()) {
        if (IsDocxExtraXmlPart(entry)) {
            xml_parts.push_back(entry);
        }
    }

    std::vector<DocumentBlock> blocks;
    std::uint64_t order = 0;
    std::size_t total_xml_bytes = 0;
    for (const auto& part : xml_parts) {
        auto xml = archive.value().ReadTextEntry(part);
        if (!xml.ok()) {
            if (part == "word/document.xml") {
                return xml.status();
            }
            continue;
        }
        total_xml_bytes += xml.value().size();
        if (total_xml_bytes > options_.max_total_uncompressed_bytes) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "docx XML payload is too large");
        }

        core::Status parse_status;
        auto doc = ParseXml(xml.value(), &parse_status);
        if (!parse_status.ok()) {
            return parse_status;
        }
        int paragraph_index = 0;
        ForEachDescendantByLocalName(doc, "p", [&](const pugi::xml_node& paragraph) {
            auto text = DocxParagraphText(paragraph);
            if (text.empty()) {
                ++paragraph_index;
                return;
            }
            blocks.push_back(DocxBlockFromParagraph(paragraph, order++, paragraph_index, part, std::move(text)));
            ++paragraph_index;
        });
    }
    return blocks;
}

core::Result<std::vector<DocumentBlock>> OoxmlExtractor::ExtractPptxBlocks(const std::filesystem::path& path) const {
    auto archive = ZipArchive::Open(path, options_);
    if (!archive.ok()) {
        return archive.status();
    }
    if (auto status = ValidateOoxmlPackage(archive.value(), "pptx", options_); !status.ok()) {
        return status;
    }
    auto entries = archive.value().ListEntries();
    if (!entries.ok()) {
        return entries.status();
    }

    std::vector<std::string> slides;
    for (const auto& entry : entries.value()) {
        if (IsPptxSlideXmlPart(entry)) {
            slides.push_back(entry);
        }
    }
    std::sort(slides.begin(), slides.end(), [](const std::string& left, const std::string& right) {
        return PptxSlideNumber(left) < PptxSlideNumber(right);
    });

    std::vector<DocumentBlock> blocks;
    std::uint64_t order = 0;
    std::size_t total_xml_bytes = 0;
    for (const auto& slide_part : slides) {
        auto xml = archive.value().ReadTextEntry(slide_part);
        if (!xml.ok()) {
            return xml.status();
        }
        total_xml_bytes += xml.value().size();
        if (total_xml_bytes > options_.max_total_uncompressed_bytes) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "pptx XML payload is too large");
        }

        core::Status parse_status;
        auto doc = ParseXml(xml.value(), &parse_status);
        if (!parse_status.ok()) {
            return parse_status;
        }

        const auto slide_no = PptxSlideNumber(slide_part);
        int shape_index = 0;
        ForEachDescendantByLocalName(doc, "sp", [&](const pugi::xml_node& shape) {
            const auto shape_kind = PptxShapeKind(shape);
            int paragraph_index = 0;
            ForEachDescendantByLocalName(shape, "p", [&](const pugi::xml_node& paragraph) {
                auto text = PptxParagraphText(paragraph);
                if (text.empty()) {
                    ++paragraph_index;
                    return;
                }

                DocumentBlock block;
                block.id = DocumentBlockId("pptx", order);
                block.text = std::move(text);
                block.source = "pptx";
                block.order = order++;
                block.kind = shape_kind;
                block.slide = slide_no;
                block.paragraph_index = paragraph_index;
                block.font_size = PptxFontSize(paragraph);
                block.confidence = shape_kind == "heading" ? 0.86 : 0.7;
                block.metadata["xml_part"] = slide_part;
                block.metadata["shape_index"] = std::to_string(shape_index);

                auto ppr = FirstChildByLocalName(paragraph, "pPr");
                if (ppr) {
                    block.bullet_level = AttrInt(ppr, "lvl").value_or(0);
                }
                if (shape_kind == "heading") {
                    block.heading_level = 1;
                } else if (auto numbered = DetectNumberingLevel(block.text)) {
                    block.kind = "heading";
                    block.heading_level = slide_no > 0 ? *numbered + 1 : *numbered;
                }
                if (block.kind != "heading") {
                    auto inferred = InferHeadingLevel(block);
                    if (inferred && block.font_size && *block.font_size >= 18.0) {
                        block.kind = "heading";
                        block.heading_level = std::min(5, *inferred);
                        block.confidence = 0.8;
                    }
                }
                blocks.push_back(std::move(block));
                ++paragraph_index;
            });
            ++shape_index;
        });
    }
    return blocks;
}

} // namespace agent::document
