#include "document_analysis_service.h"
#include "document_file_store.h"
#include "document_metadata_repository.h"
#include "ooxml_extractor.h"
#include "semantic_cache_pipeline.h"
#include "sqlite/sqlite_connection_pool.h"
#include "sqlite/sqlite_statement.h"

#include <gtest/gtest.h>
#include <zip.h>

#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace {

using agent::document::DetectNumberingLevel;
using agent::document::AnalyzeDocument;
using agent::document::BuildMindmap;
using agent::document::BuildDiagnosis;
using agent::document::BuildLocalChunks;
using agent::document::ChunkSlice;
using agent::document::ChunkTrunk;
using agent::document::DocumentAnalysisOptions;
using agent::document::DocumentAnalysisService;
using agent::document::DocumentAnalyzeRequest;
using agent::document::DocumentAnalyzeResponse;
using agent::document::DocumentBlock;
using agent::document::DocumentFileStore;
using agent::document::DocumentFileStoreOptions;
using agent::document::DocumentMetadataRecord;
using agent::document::DocumentMetadataRepository;
using agent::document::HeadingLevelFromStyle;
using agent::document::IDocumentEmbeddingProvider;
using agent::document::OoxmlExtractor;

class FakeDocumentSemanticCache final : public agent::semantic_cache::ISemanticCache {
public:
    core::Result<agent::semantic_cache::CacheLookupResult> Lookup(
        const agent::semantic_cache::CacheLookupRequest& req) override {
        ++lookup_count;
        last_lookup = req;
        agent::semantic_cache::CacheLookupResult result;
        result.hit = hit;
        result.similarity_score = score;
        result.payload = payload;
        return result;
    }

    core::Status Store(const agent::semantic_cache::CacheStoreRequest& req) override {
        ++store_count;
        last_store = req;
        return core::Status::Ok();
    }

    bool hit = false;
    float score = 0.95f;
    std::string payload;
    int lookup_count = 0;
    int store_count = 0;
    agent::semantic_cache::CacheLookupRequest last_lookup;
    agent::semantic_cache::CacheStoreRequest last_store;
};

class FakeChunkLlmClient final : public agent::llm::ILlmClient {
public:
    core::Result<agent::llm::ChatCompletionResponse> Complete(
        const agent::llm::ChatCompletionRequest& req) override {
        ++call_count;
        last_request = req;
        agent::llm::ChatCompletionResponse response;
        response.model = "fake-chunk-model";
        response.content = R"({"title":"函数关系","summary":"函数输入输出关系概括","slices":[{"title":"概念","summary":"函数关系说明","text":"函数表示输入与输出之间的对应关系","kind":"concept","confidence":0.88}]})";
        return response;
    }

    agent::llm::ChatCompletionRequest last_request;
    int call_count = 0;
};

class FakeEmbeddingProvider final : public IDocumentEmbeddingProvider {
public:
    explicit FakeEmbeddingProvider(std::size_t dim) : dim_(dim) {}

    core::Result<std::vector<float>> EmbedText(std::string_view text) override {
        std::vector<float> embedding(dim_, 0.0f);
        if (embedding.empty()) {
            return embedding;
        }
        if (text.find("geometry") != std::string_view::npos) {
            embedding[1 % dim_] = 1.0f;
        } else {
            embedding[0] = 1.0f;
        }
        return embedding;
    }

private:
    std::size_t dim_;
};

void AddZipText(zip_t* archive, const char* name, const std::string& text) {
    auto* source = zip_source_buffer(archive, text.data(), text.size(), 0);
    ASSERT_NE(source, nullptr);
    ASSERT_GE(zip_file_add(archive, name, source, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8), 0);
}

std::filesystem::path TempPath(const std::string& name) {
    return std::filesystem::temp_directory_path() / name;
}

void WriteZip(const std::filesystem::path& path, const std::vector<std::pair<std::string, std::string>>& entries) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    int error = 0;
    zip_t* archive = zip_open(path.string().c_str(), ZIP_CREATE | ZIP_TRUNCATE, &error);
    ASSERT_NE(archive, nullptr);
    const auto has_content_types = std::any_of(entries.begin(), entries.end(), [](const auto& entry) {
        return entry.first == "[Content_Types].xml";
    });
    if (!has_content_types) {
        AddZipText(archive, "[Content_Types].xml", R"(<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
</Types>)");
    }
    for (const auto& [name, text] : entries) {
        AddZipText(archive, name.c_str(), text);
    }
    ASSERT_EQ(zip_close(archive), 0);
}

std::string ReadBinaryFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::string data;
    input.seekg(0, std::ios::end);
    data.resize(static_cast<std::size_t>(input.tellg()));
    input.seekg(0, std::ios::beg);
    if (!data.empty()) {
        input.read(data.data(), static_cast<std::streamsize>(data.size()));
    }
    return data;
}

void WriteBinaryFile(const std::filesystem::path& path, std::string_view data) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output.good());
    output.write(data.data(), static_cast<std::streamsize>(data.size()));
}

std::string MinimalDocxXml() {
    return R"(<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>第一章 函数基础</w:t></w:r>
    </w:p>
    <w:p>
      <w:r><w:rPr><w:b/><w:sz w:val="28"/></w:rPr><w:t>核心概念</w:t></w:r>
    </w:p>
    <w:p>
      <w:r><w:t>例题：求一次函数的解析式。</w:t></w:r>
    </w:p>
  </w:body>
</w:document>)";
}

std::string DocxXmlWithParagraph(std::string_view text) {
    return std::string(R"(<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>)") + std::string(text) + R"(</w:t></w:r></w:p>
  </w:body>
</w:document>)";
}

std::string MinimalPptxSlideXml(int slide) {
    return R"(<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld><p:spTree>
    <p:sp>
      <p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
      <p:txBody><a:p><a:r><a:t>课堂导入</a:t></a:r></a:p></p:txBody>
    </p:sp>
    <p:sp>
      <p:txBody>
        <a:p><a:pPr lvl="1"/><a:r><a:rPr sz="2400"/><a:t>1.1 观察图像变化</a:t></a:r></a:p>
        <a:p><a:r><a:t>练习：判断函数单调性。</a:t></a:r></a:p>
      </p:txBody>
    </p:sp>
  </p:spTree></p:cSld>
</p:sld>)";
}

} // namespace

TEST(DocumentOoxmlUtilsTest, InfersHeadingSignals) {
    EXPECT_EQ(HeadingLevelFromStyle("Heading2"), 2);
    EXPECT_EQ(HeadingLevelFromStyle("标题三"), 3);
    EXPECT_EQ(DetectNumberingLevel("1.2.3 分层目标"), 3);
    EXPECT_EQ(DetectNumberingLevel("（一）核心概念"), 2);
}

TEST(DocumentOoxmlExtractorTest, ExtractsDocxBlocksFromOoxmlPackage) {
    const auto path = TempPath("agent_document_test.docx");
    WriteZip(path, {{"word/document.xml", MinimalDocxXml()}});

    OoxmlExtractor extractor;
    auto blocks = extractor.ExtractDocxBlocks(path);
    ASSERT_TRUE(blocks.ok()) << blocks.status().message();
    ASSERT_EQ(blocks.value().size(), 3u);

    EXPECT_EQ(blocks.value()[0].id, "docx-0");
    EXPECT_EQ(blocks.value()[0].text, "第一章 函数基础");
    EXPECT_EQ(blocks.value()[0].kind, "heading");
    ASSERT_TRUE(blocks.value()[0].heading_level);
    EXPECT_EQ(*blocks.value()[0].heading_level, 1);

    EXPECT_EQ(blocks.value()[1].text, "核心概念");
    EXPECT_EQ(blocks.value()[1].kind, "heading");
    ASSERT_TRUE(blocks.value()[1].font_size);
    EXPECT_DOUBLE_EQ(*blocks.value()[1].font_size, 14.0);

    EXPECT_EQ(blocks.value()[2].text, "例题：求一次函数的解析式。");
    EXPECT_EQ(blocks.value()[2].kind, "paragraph");

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(DocumentOoxmlSecurityTest, RejectsIncompleteOoxmlPackageWithoutContentTypes) {
    const auto path = TempPath("agent_document_incomplete.docx");
    std::error_code ec;
    std::filesystem::remove(path, ec);
    int error = 0;
    zip_t* archive = zip_open(path.string().c_str(), ZIP_CREATE | ZIP_TRUNCATE, &error);
    ASSERT_NE(archive, nullptr);
    AddZipText(archive, "word/document.xml", MinimalDocxXml());
    ASSERT_EQ(zip_close(archive), 0);

    OoxmlExtractor extractor;
    auto blocks = extractor.ExtractDocxBlocks(path);
    EXPECT_FALSE(blocks.ok());

    std::filesystem::remove(path, ec);
}

TEST(DocumentOoxmlSecurityTest, RejectsSelfExtractingZipPrefix) {
    const auto valid_path = TempPath("agent_document_valid_for_sfx.docx");
    const auto sfx_path = TempPath("agent_document_sfx_prefix.docx");
    WriteZip(valid_path, {{"word/document.xml", MinimalDocxXml()}});

    std::string sfx_payload = "MZ";
    sfx_payload.append(256, '\0');
    sfx_payload.append(ReadBinaryFile(valid_path));
    WriteBinaryFile(sfx_path, sfx_payload);

    OoxmlExtractor extractor;
    auto blocks = extractor.ExtractDocxBlocks(sfx_path);
    ASSERT_FALSE(blocks.ok());
    EXPECT_EQ(blocks.status().code(), core::ErrorCode::PermissionDenied);
    EXPECT_NE(blocks.status().message().find("self-extracting"), std::string::npos);

    std::error_code ec;
    std::filesystem::remove(valid_path, ec);
    std::filesystem::remove(sfx_path, ec);
}

TEST(DocumentOoxmlSecurityTest, RejectsDocxWithMacroProject) {
    const auto path = TempPath("agent_document_macro.docx");
    WriteZip(path, {
        {"word/document.xml", MinimalDocxXml()},
        {"word/vbaProject.bin", "macro-binary"},
    });

    OoxmlExtractor extractor;
    auto blocks = extractor.ExtractDocxBlocks(path);
    ASSERT_FALSE(blocks.ok());
    EXPECT_EQ(blocks.status().code(), core::ErrorCode::PermissionDenied);
    EXPECT_NE(blocks.status().message().find("macro"), std::string::npos);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(DocumentOoxmlSecurityTest, RejectsPptxWithEmbeddedExecutablePackage) {
    const auto path = TempPath("agent_document_embedded_exec.pptx");
    WriteZip(path, {
        {"ppt/slides/slide1.xml", MinimalPptxSlideXml(1)},
        {"ppt/embeddings/payload.exe", "MZ"},
    });

    OoxmlExtractor extractor;
    auto blocks = extractor.ExtractPptxBlocks(path);
    ASSERT_FALSE(blocks.ok());
    EXPECT_EQ(blocks.status().code(), core::ErrorCode::PermissionDenied);
    EXPECT_NE(blocks.status().message().find("embedded"), std::string::npos);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(DocumentOoxmlExtractorTest, ExtractsPptxBlocksBySlideOrder) {
    const auto path = TempPath("agent_document_test.pptx");
    WriteZip(path, {
        {"ppt/slides/slide2.xml", MinimalPptxSlideXml(2)},
        {"ppt/slides/slide1.xml", MinimalPptxSlideXml(1)},
    });

    OoxmlExtractor extractor;
    auto blocks = extractor.ExtractPptxBlocks(path);
    ASSERT_TRUE(blocks.ok()) << blocks.status().message();
    ASSERT_EQ(blocks.value().size(), 6u);

    EXPECT_EQ(blocks.value()[0].id, "pptx-0");
    EXPECT_EQ(blocks.value()[0].text, "课堂导入");
    EXPECT_EQ(blocks.value()[0].kind, "heading");
    ASSERT_TRUE(blocks.value()[0].slide);
    EXPECT_EQ(*blocks.value()[0].slide, 1);

    EXPECT_EQ(blocks.value()[1].text, "1.1 观察图像变化");
    EXPECT_EQ(blocks.value()[1].kind, "heading");
    ASSERT_TRUE(blocks.value()[1].heading_level);
    EXPECT_EQ(*blocks.value()[1].heading_level, 3);

    EXPECT_EQ(blocks.value()[2].text, "练习：判断函数单调性。");
    EXPECT_EQ(blocks.value()[2].kind, "paragraph");
    ASSERT_TRUE(blocks.value()[3].slide);
    EXPECT_EQ(*blocks.value()[3].slide, 2);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(DocumentMindmapBuilderTest, BuildsHeadingTreeAndUnclassifiedContent) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "b0", .text = "课前说明", .source = "docx", .order = 0});
    blocks.push_back(DocumentBlock{.id = "b1", .text = "第一章 函数基础", .source = "docx", .order = 1, .kind = "heading", .heading_level = 1});
    blocks.push_back(DocumentBlock{.id = "b2", .text = "函数表示变量之间的对应关系。", .source = "docx", .order = 2});
    blocks.push_back(DocumentBlock{.id = "b3", .text = "1.1 一次函数", .source = "docx", .order = 3, .kind = "heading", .heading_level = 2});
    blocks.push_back(DocumentBlock{.id = "b4", .text = "例题：求一次函数解析式。", .source = "docx", .order = 4});

    auto mindmap = BuildMindmap(blocks, "lesson.docx");
    ASSERT_EQ(mindmap["name"], "lesson");
    ASSERT_EQ(mindmap["children"].size(), 2u);
    EXPECT_EQ(mindmap["children"][0]["name"], "Unclassified Content");
    EXPECT_EQ(mindmap["children"][0]["children"][0]["name"], "课前说明");
    EXPECT_EQ(mindmap["children"][1]["name"], "第一章 函数基础");
    EXPECT_EQ(mindmap["children"][1]["children"][0]["name"], "函数表示变量之间的对应关系。");
    EXPECT_EQ(mindmap["children"][1]["children"][1]["name"], "1.1 一次函数");
    EXPECT_EQ(mindmap["children"][1]["children"][1]["children"][0]["name"], "例题：求一次函数解析式。");
}

TEST(DocumentMindmapBuilderTest, AttachesChunkToNearestScopedHeading) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "h1", .text = "函数基础", .source = "pptx", .order = 0, .kind = "heading", .slide = 1, .heading_level = 1});
    blocks.push_back(DocumentBlock{.id = "w1", .text = "函数输入输出关系说明", .source = "pptx", .order = 1, .slide = 1, .confidence = 0.5});
    blocks.push_back(DocumentBlock{.id = "h2", .text = "几何图形", .source = "pptx", .order = 2, .kind = "heading", .slide = 2, .heading_level = 1});

    ChunkTrunk chunk;
    chunk.chunk_id = "chunk-1";
    chunk.block_ids = {"w1"};
    chunk.text = "函数输入输出关系说明";
    chunk.title = "函数关系";
    chunk.summary = "输入和输出之间的对应关系";
    chunk.order = 1;
    chunk.slide = 1;
    chunk.slices = {ChunkSlice{.title = "对应关系", .summary = "变量对应", .text = "函数输入输出关系说明", .kind = "concept", .confidence = 0.7}};

    auto mindmap = BuildMindmap(blocks, "slides.pptx", {chunk});
    ASSERT_EQ(mindmap["children"].size(), 2u);
    EXPECT_EQ(mindmap["children"][0]["name"], "函数基础");
    ASSERT_EQ(mindmap["children"][0]["children"].size(), 1u);
    EXPECT_EQ(mindmap["children"][0]["children"][0]["name"], "函数关系");
    EXPECT_EQ(mindmap["children"][0]["children"][0]["children"][0]["name"], "对应关系");
    EXPECT_EQ(mindmap["children"][1]["name"], "几何图形");
}

TEST(DocumentMindmapBuilderTest, UsesEmbeddingProviderForChunkParentSelection) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "h1", .text = "函数基础", .source = "docx", .order = 0, .kind = "heading", .heading_level = 1});
    blocks.push_back(DocumentBlock{.id = "h2", .text = "geometry shapes", .source = "docx", .order = 10, .kind = "heading", .heading_level = 1});

    ChunkTrunk chunk;
    chunk.chunk_id = "chunk-geometry";
    chunk.block_ids = {"w1"};
    chunk.text = "geometry provider concept";
    chunk.title = "geometry concept";
    chunk.summary = "geometry provider concept";
    chunk.order = 1;
    chunk.slices = {ChunkSlice{.title = "geometry slice", .summary = "geometry", .text = "geometry provider concept", .kind = "concept", .confidence = 0.8}};

    auto provider = std::make_shared<FakeEmbeddingProvider>(agent::semantic_cache::kExpectedEmbeddingDim);
    auto mindmap = BuildMindmap(blocks, "lesson.docx", {chunk}, provider);
    ASSERT_EQ(mindmap["children"].size(), 2u);
    EXPECT_EQ(mindmap["children"][0]["name"], "函数基础");
    EXPECT_TRUE(mindmap["children"][0]["children"].empty());
    EXPECT_EQ(mindmap["children"][1]["name"], "geometry shapes");
    ASSERT_EQ(mindmap["children"][1]["children"].size(), 1u);
    EXPECT_EQ(mindmap["children"][1]["children"][0]["name"], "geometry concept");
}

TEST(DocumentMindmapBuilderTest, FiltersSeparatorOnlyNodes) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "h1", .text = "课程结构", .source = "docx", .order = 0, .kind = "heading", .heading_level = 1});
    blocks.push_back(DocumentBlock{.id = "sep", .text = "---", .source = "docx", .order = 1});
    blocks.push_back(DocumentBlock{.id = "b1", .text = "学习目标：理解函数关系。", .source = "docx", .order = 2});

    ChunkTrunk chunk;
    chunk.chunk_id = "chunk-1";
    chunk.block_ids = {"b1"};
    chunk.text = "学习目标：理解函数关系。";
    chunk.title = "函数关系";
    chunk.summary = "理解函数关系";
    chunk.order = 2;
    chunk.slices = {
        ChunkSlice{.title = "---", .summary = "---", .text = "---", .kind = "paragraph", .confidence = 0.1},
        ChunkSlice{.title = "学习目标", .summary = "理解函数关系", .text = "学习目标：理解函数关系。", .kind = "objective", .confidence = 0.8},
    };

    auto mindmap = BuildMindmap(blocks, "lesson.docx", {chunk});
    ASSERT_EQ(mindmap["children"].size(), 1u);
    ASSERT_EQ(mindmap["children"][0]["children"].size(), 1u);
    const auto& chunk_node = mindmap["children"][0]["children"][0];
    EXPECT_EQ(chunk_node["name"], "函数关系");
    ASSERT_EQ(chunk_node["children"].size(), 1u);
    EXPECT_EQ(chunk_node["children"][0]["name"], "学习目标");
}

TEST(DocumentChunkBuilderTest, BuildsLocalChunksForWeakUnstructuredBlocks) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "h1", .text = "函数基础", .source = "docx", .order = 0, .kind = "heading", .heading_level = 1});
    blocks.push_back(DocumentBlock{
        .id = "b1",
        .text = "目标：理解函数输入与输出之间的对应关系。例题：根据图像判断函数单调性。练习：完成课后检测题。",
        .source = "docx",
        .order = 1,
        .confidence = 0.5});

    auto chunks = BuildLocalChunks(blocks);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].chunk_id, "chunk-0");
    EXPECT_EQ(chunks[0].block_ids, (std::vector<std::string>{"h1", "b1"}));
    EXPECT_EQ(chunks[0].source, "local");
    ASSERT_GE(chunks[0].slices.size(), 4u);
    EXPECT_EQ(chunks[0].slices[0].kind, "concept");
    EXPECT_EQ(chunks[0].slices[1].kind, "objective");
    EXPECT_EQ(chunks[0].slices[2].kind, "example");
    EXPECT_EQ(chunks[0].slices[3].kind, "practice");
}

TEST(DocumentChunkBuilderTest, GroupsAdjacentWeakBlocksWithoutLocalHashEmbedding) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "b1", .text = "函数 输入 输出 对应 关系 变量", .source = "docx", .order = 1, .confidence = 0.5});
    blocks.push_back(DocumentBlock{.id = "b2", .text = "函数 输入 输出 对应 关系 图像", .source = "docx", .order = 2, .confidence = 0.5});
    blocks.push_back(DocumentBlock{.id = "b3", .text = "三角形 面积 高 底边 几何 图形", .source = "docx", .order = 3, .confidence = 0.5});

    DocumentAnalysisOptions options;
    options.chunk_similarity_threshold = 0.6;
    auto chunks = BuildLocalChunks(blocks, options);
    ASSERT_EQ(chunks.size(), 1u);
    const std::vector<std::string> ids{"b1", "b2", "b3"};
    EXPECT_EQ(chunks[0].block_ids, ids);
}

TEST(DocumentChunkBuilderTest, UsesSupportedEmbeddingProviderDimensionsForClustering) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "b1", .text = "alpha provider concept", .source = "docx", .order = 1, .confidence = 0.5});
    blocks.push_back(DocumentBlock{.id = "b2", .text = "beta provider concept", .source = "docx", .order = 2, .confidence = 0.5});
    blocks.push_back(DocumentBlock{.id = "b3", .text = "geometry provider concept", .source = "docx", .order = 3, .confidence = 0.5});

    DocumentAnalysisOptions options;
    options.chunk_similarity_threshold = 0.9;
    auto provider = std::make_shared<FakeEmbeddingProvider>(agent::semantic_cache::kExpectedEmbeddingDim);

    auto chunks = BuildLocalChunks(blocks, options, {}, provider);
    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks[0].block_ids, (std::vector<std::string>{"b1", "b2"}));
    EXPECT_EQ(chunks[1].block_ids, (std::vector<std::string>{"b3"}));
}

TEST(DocumentChunkBuilderTest, IgnoresUnsupportedProviderDimensionsWithoutHashEmbedding) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "b1", .text = "same same same", .source = "docx", .order = 1, .confidence = 0.5});
    blocks.push_back(DocumentBlock{.id = "b2", .text = "same same same", .source = "docx", .order = 2, .confidence = 0.5});

    DocumentAnalysisOptions options;
    options.chunk_similarity_threshold = 0.9;
    auto unsupported_provider = std::make_shared<FakeEmbeddingProvider>(7);

    auto chunks = BuildLocalChunks(blocks, options, {}, unsupported_provider);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].block_ids, (std::vector<std::string>{"b1", "b2"}));
}

TEST(DocumentChunkBuilderTest, AttachesNumberedTitleToFollowingWeakBlock) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "h1", .text = "（1）系统文献调研与关键技术储备", .source = "docx", .order = 1, .confidence = 0.9});
    blocks.push_back(DocumentBlock{.id = "b1", .text = "围绕对抗性领域适应和跨模态注意力机制阅读文献，总结经验并寻找创新空间。", .source = "docx", .order = 2, .confidence = 0.5});

    auto chunks = BuildLocalChunks(blocks);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].block_ids, (std::vector<std::string>{"h1", "b1"}));
    EXPECT_NE(chunks[0].text.find("系统文献调研"), std::string::npos);
    EXPECT_NE(chunks[0].text.find("对抗性领域适应"), std::string::npos);
}

TEST(DocumentChunkBuilderTest, UsesLlmFallbackWhenEnabled) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "b1", .text = "函数 输入 输出 对应 关系", .source = "docx", .order = 1, .confidence = 0.5});

    DocumentAnalysisOptions options;
    options.enable_llm_chunk_fallback = true;
    options.chunk_llm_model = "chunk-model";
    auto llm = std::make_shared<FakeChunkLlmClient>();
    auto chunks = BuildLocalChunks(blocks, options, llm);
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].source, "llm");
    EXPECT_EQ(chunks[0].title, "函数关系");
    ASSERT_EQ(chunks[0].slices.size(), 1u);
    EXPECT_EQ(chunks[0].slices[0].kind, "concept");
    EXPECT_EQ(llm->last_request.model, "chunk-model");
}

TEST(DocumentChunkBuilderTest, UsesSemanticCacheBeforeLlmFallback) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "b1", .text = "函数 输入 输出 对应 关系", .source = "docx", .order = 1, .confidence = 0.5});

    DocumentAnalysisOptions options;
    options.enable_llm_chunk_fallback = true;
    options.chunk_llm_model = "chunk-model";

    auto llm = std::make_shared<FakeChunkLlmClient>();
    auto semantic_cache = std::make_shared<FakeDocumentSemanticCache>();
    semantic_cache->hit = true;
    semantic_cache->payload =
        R"({"title":"缓存标题","summary":"缓存摘要","slices":[{"title":"缓存切片","summary":"切片摘要","text":"函数表示输入与输出之间的对应关系","kind":"concept","confidence":0.9}],"confidence":0.8,"metadata":{}})";

    agent::document::DocumentChunkBuildMetrics metrics;
    auto chunks = BuildLocalChunks(
        blocks,
        options,
        llm,
        nullptr,
        nullptr,
        semantic_cache,
        &metrics);

    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].title, "缓存标题");
    EXPECT_EQ(chunks[0].summary, "缓存摘要");
    EXPECT_TRUE(chunks[0].reused);
    EXPECT_EQ(chunks[0].metadata["semantic_cache_hit"], "true");
    EXPECT_EQ(llm->call_count, 0);
    EXPECT_EQ(semantic_cache->lookup_count, 1);
    EXPECT_EQ(metrics.semantic_cache_hit_count, 1u);
    EXPECT_EQ(metrics.llm_direct_count, 0u);
    EXPECT_EQ(semantic_cache->last_lookup.topic, "document_chunk_title");
}

TEST(DocumentAnalysisPipelineTest, EmitsLocalSemanticChunks) {
    const auto path = TempPath("agent_document_chunks.docx");
    WriteZip(path, {{"word/document.xml", R"(<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>函数基础</w:t></w:r>
    </w:p>
    <w:p>
      <w:r><w:t>目标：理解函数输入与输出之间的对应关系。例题：根据图像判断函数单调性。练习：完成课后检测题。</w:t></w:r>
    </w:p>
  </w:body>
</w:document>)"}});

    auto result = AnalyzeDocument(path, "chunked.docx");
    ASSERT_TRUE(result.ok()) << result.status().message();
    ASSERT_EQ(result.value()["chunks"].size(), 1u);
    EXPECT_EQ(result.value()["chunks"][0]["source"], "local");
    EXPECT_EQ(result.value()["runNodes"][1]["output"]["status"], "local");
    EXPECT_EQ(result.value()["runNodes"][1]["output"]["chunkCount"], 1);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(DocumentDiagnosisBuilderTest, ScoresStructuredTeachingMaterial) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "b0", .text = "学习目标：理解一次函数。", .source = "docx", .order = 0, .kind = "heading", .heading_level = 1});
    blocks.push_back(DocumentBlock{.id = "b1", .text = "重点：函数表达式和图像。", .source = "docx", .order = 1, .kind = "heading", .heading_level = 2});
    blocks.push_back(DocumentBlock{.id = "b2", .text = "例题：求一次函数解析式。", .source = "docx", .order = 2});
    blocks.push_back(DocumentBlock{.id = "b3", .text = "练习：判断函数单调性。", .source = "docx", .order = 3});

    ChunkTrunk chunk;
    chunk.chunk_id = "chunk-1";
    chunk.title = "函数应用";
    chunk.summary = "一次函数应用";
    chunk.order = 2;

    auto mindmap = BuildMindmap(blocks, "lesson.docx", {chunk});
    auto diagnosis = BuildDiagnosis(blocks, mindmap, {chunk});
    EXPECT_GE(diagnosis["score"].get<double>(), 7.0);
    EXPECT_EQ(diagnosis["rule_features"]["has_examples"], true);
    EXPECT_EQ(diagnosis["rule_features"]["has_practice"], true);
    EXPECT_EQ(diagnosis["rule_features"]["has_objective"], true);
    EXPECT_TRUE(diagnosis["suggestions"].back().get<std::string>().find("课堂模拟") != std::string::npos);
}

TEST(DocumentDiagnosisBuilderTest, SuggestsMissingTeachingElements) {
    std::vector<DocumentBlock> blocks;
    blocks.push_back(DocumentBlock{.id = "b0", .text = "函数基础", .source = "docx", .order = 0, .kind = "heading", .heading_level = 1});
    blocks.push_back(DocumentBlock{.id = "b1", .text = "函数表示两个变量之间的关系。", .source = "docx", .order = 1});

    auto mindmap = BuildMindmap(blocks, "lesson.docx");
    auto diagnosis = BuildDiagnosis(blocks, mindmap);
    EXPECT_LT(diagnosis["score"].get<double>(), 7.0);
    EXPECT_EQ(diagnosis["rule_features"]["has_objective"], false);
    EXPECT_EQ(diagnosis["rule_features"]["has_practice"], false);
    ASSERT_GE(diagnosis["suggestions"].size(), 3u);
    EXPECT_NE(diagnosis["suggestions"][0].get<std::string>().find("学习目标"), std::string::npos);
    EXPECT_NE(diagnosis["suggestions"][1].get<std::string>().find("例题"), std::string::npos);
    EXPECT_NE(diagnosis["knowledge_coverage"]["enabled"].get<bool>(), true);
}

TEST(DocumentAnalysisPipelineTest, AnalyzesDocxIntoResultSchema) {
    const auto path = TempPath("agent_document_pipeline.docx");
    WriteZip(path, {{"word/document.xml", MinimalDocxXml()}});

    auto result = AnalyzeDocument(path, "teaching-plan.docx");
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_EQ(result.value()["fileName"], "teaching-plan.docx");
    EXPECT_EQ(result.value()["fileType"], "docx");
    EXPECT_EQ(result.value()["schemaVersion"], "document_analysis.v1");
    EXPECT_EQ(result.value()["blocks"].size(), 3u);
    EXPECT_TRUE(result.value()["mindmap"].contains("children"));
    EXPECT_TRUE(result.value()["diagnosis"].contains("score"));
    EXPECT_GE(result.value()["tokenCount"].get<std::uint64_t>(), 1u);
    ASSERT_GE(result.value()["runNodes"].size(), 4u);
    EXPECT_EQ(result.value()["runNodes"][0]["nodeName"], "参数提取2");

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(DocumentAnalysisServiceTest, RunsAnalyzeOnComputePoolAndInvokesCallback) {
    const auto path = TempPath("agent_document_async.docx");
    WriteZip(path, {{"word/document.xml", MinimalDocxXml()}});

    core::ThreadPool compute({1, 8, "document-compute"});
    core::ThreadPool io({1, 8, "document-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    DocumentAnalysisService service(compute, io);
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    core::Result<DocumentAnalyzeResponse> captured(
        core::Status::Error(core::ErrorCode::Unknown, "not completed"));

    DocumentAnalyzeRequest request;
    request.path = path;
    request.file_name = "async-lesson.docx";
    request.trace_id = "trace-document-async";

    auto submit = service.SubmitAnalyze(std::move(request), [&](core::Result<DocumentAnalyzeResponse> result) {
        {
            std::lock_guard lock(mutex);
            captured = std::move(result);
            done = true;
        }
        cv.notify_one();
    });
    ASSERT_TRUE(submit.ok()) << submit.message();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(5), [&] { return done; }));
    }

    ASSERT_TRUE(captured.ok()) << captured.status().message();
    EXPECT_EQ(captured.value().trace_id, "trace-document-async");
    EXPECT_EQ(captured.value().result["fileName"], "async-lesson.docx");
    EXPECT_EQ(captured.value().result["fileType"], "docx");
    EXPECT_EQ(captured.value().result["schemaVersion"], "document_analysis.v1");
    EXPECT_GE(captured.value().latency.total.count(), 0);

    compute.Shutdown(true);
    io.Shutdown(true);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(DocumentAnalysisServiceTest, StoresAnalyzeResultOnIoPoolWhenRepositoryIsConfigured) {
    const auto document_path = TempPath("agent_document_store.docx");
    const auto db_path = TempPath("agent_document_store.sqlite");
    WriteZip(document_path, {{"word/document.xml", MinimalDocxXml()}});
    std::error_code ec;
    std::filesystem::remove(db_path, ec);

    core::ThreadPool compute({1, 8, "document-compute"});
    core::ThreadPool io({1, 8, "document-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    storage::sqlite::SqliteConnectionPoolOptions pool_options;
    pool_options.path = db_path.string();
    pool_options.read_connection_count = 1;
    pool_options.write_connection_count = 1;
    pool_options.busy_timeout_ms = 250;
    auto repository = std::make_shared<storage::sqlite::SqliteConnectionPool>(pool_options);
    ASSERT_TRUE(repository->Start().ok());

    DocumentAnalysisService service(compute, io);
    ASSERT_TRUE(service.SetRepository(repository).ok());

    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    core::Result<DocumentAnalyzeResponse> captured(
        core::Status::Error(core::ErrorCode::Unknown, "not completed"));

    DocumentAnalyzeRequest request;
    request.path = document_path;
    request.file_name = "stored-lesson.docx";
    request.trace_id = "trace-document-store";
    request.document_id = "document-store-1";

    auto submit = service.SubmitAnalyze(std::move(request), [&](core::Result<DocumentAnalyzeResponse> result) {
        {
            std::lock_guard lock(mutex);
            captured = std::move(result);
            done = true;
        }
        cv.notify_one();
    });
    ASSERT_TRUE(submit.ok()) << submit.message();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(10), [&] { return done; }));
    }
    ASSERT_TRUE(captured.ok()) << captured.status().message();
    EXPECT_EQ(captured.value().document_id, "document-store-1");

    auto lease_result = repository->WaitAcquireReadFor(std::chrono::seconds(5));
    ASSERT_TRUE(lease_result.ok()) << lease_result.status().message();
    auto lease = std::move(lease_result).value();
    auto stmt_result = lease.connection().Prepare(
        "SELECT trace_id, file_name, file_type, schema_version, result_json "
        "FROM document_analysis_results WHERE document_id = ?1");
    ASSERT_TRUE(stmt_result.ok()) << stmt_result.status().message();
    auto stmt = std::move(stmt_result).value();
    ASSERT_TRUE(stmt.BindText(1, "document-store-1").ok());
    auto step = stmt.Step();
    ASSERT_TRUE(step.ok()) << step.status().message();
    ASSERT_EQ(step.value(), storage::sqlite::SqliteStepResult::Row);
    EXPECT_EQ(stmt.ColumnText(0), "trace-document-store");
    EXPECT_EQ(stmt.ColumnText(1), "stored-lesson.docx");
    EXPECT_EQ(stmt.ColumnText(2), "docx");
    EXPECT_EQ(stmt.ColumnText(3), "document_analysis.v1");
    auto payload = nlohmann::json::parse(stmt.ColumnText(4));
    EXPECT_EQ(payload["fileName"], "stored-lesson.docx");
    EXPECT_TRUE(payload.contains("diagnosis"));

    repository->Close();
    compute.Shutdown(true);
    io.Shutdown(true);
    std::filesystem::remove(document_path, ec);
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path.string() + "-wal", ec);
    std::filesystem::remove(db_path.string() + "-shm", ec);
}

TEST(DocumentMetadataRepositoryTest, UpsertsAndReadsDocumentMetadata) {
    const auto db_path = TempPath("agent_document_metadata.sqlite");
    std::error_code ec;
    std::filesystem::remove(db_path, ec);

    storage::sqlite::SqliteConnectionPoolOptions pool_options;
    pool_options.path = db_path.string();
    pool_options.read_connection_count = 1;
    pool_options.write_connection_count = 1;
    pool_options.busy_timeout_ms = 250;
    auto pool = std::make_shared<storage::sqlite::SqliteConnectionPool>(pool_options);
    ASSERT_TRUE(pool->Start().ok());

    DocumentMetadataRepository repository(pool);
    ASSERT_TRUE(repository.EnsureSchema().ok());

    DocumentMetadataRecord record;
    record.document_id = "doc_meta_1";
    record.content_hash = "hash1";
    record.owner_user_uuid = "user-a";
    record.session_id = "session-a";
    record.file_name = "课程.docx";
    record.file_type = "docx";
    record.storage_path = "doc_meta_1.docx";
    record.uploaded_at_ms = 100;
    record.last_accessed_at_ms = 100;
    record.analysis_status = "uploaded";
    record.size_bytes = 42;
    ASSERT_TRUE(repository.Upsert(record).ok());

    auto loaded = repository.GetByDocumentId("doc_meta_1");
    ASSERT_TRUE(loaded.ok()) << loaded.status().message();
    EXPECT_EQ(loaded.value().file_name, "课程.docx");
    EXPECT_EQ(loaded.value().owner_user_uuid, "user-a");
    EXPECT_EQ(loaded.value().analysis_status, "uploaded");

    ASSERT_TRUE(repository.MarkAnalyzing("doc_meta_1", "trace-a").ok());
    ASSERT_TRUE(repository.MarkAnalyzed("doc_meta_1", "trace-b", 200).ok());
    loaded = repository.GetByDocumentId("doc_meta_1");
    ASSERT_TRUE(loaded.ok()) << loaded.status().message();
    EXPECT_EQ(loaded.value().analysis_status, "completed");
    EXPECT_EQ(loaded.value().analysis_trace_id, "trace-b");
    EXPECT_EQ(loaded.value().last_analyzed_at_ms, 200);

    pool->Close();
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path.string() + "-wal", ec);
    std::filesystem::remove(db_path.string() + "-shm", ec);
}

TEST(DocumentAnalysisServiceTest, ResolvesManagedDocumentByDocumentId) {
    const auto source_path = TempPath("agent_document_managed_source.docx");
    const auto db_path = TempPath("agent_document_managed.sqlite");
    const auto store_root = TempPath("agent_document_managed_store");
    WriteZip(source_path, {{"word/document.xml", MinimalDocxXml()}});
    std::error_code ec;
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove_all(store_root, ec);

    core::ThreadPool compute({1, 8, "document-compute"});
    core::ThreadPool io({1, 8, "document-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    storage::sqlite::SqliteConnectionPoolOptions pool_options;
    pool_options.path = db_path.string();
    pool_options.read_connection_count = 1;
    pool_options.write_connection_count = 1;
    pool_options.busy_timeout_ms = 250;
    auto repository = std::make_shared<storage::sqlite::SqliteConnectionPool>(pool_options);
    ASSERT_TRUE(repository->Start().ok());

    auto file_store = std::make_shared<DocumentFileStore>(DocumentFileStoreOptions{store_root});
    DocumentAnalysisService service(compute, io);
    ASSERT_TRUE(service.SetRepository(repository).ok());
    ASSERT_TRUE(service.SetFileStore(file_store).ok());

    auto imported = service.ImportManagedFile(source_path, "托管课程.docx", "user-managed", "session-managed");
    ASSERT_TRUE(imported.ok()) << imported.status().message();
    EXPECT_FALSE(imported.value().document_id.empty());
    EXPECT_TRUE(std::filesystem::exists(store_root / imported.value().storage_path));

    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    core::Result<DocumentAnalyzeResponse> captured(
        core::Status::Error(core::ErrorCode::Unknown, "not completed"));

    DocumentAnalyzeRequest request;
    request.document_id = imported.value().document_id;
    request.trace_id = "trace-managed-document";

    auto submit = service.SubmitAnalyze(std::move(request), [&](core::Result<DocumentAnalyzeResponse> result) {
        {
            std::lock_guard lock(mutex);
            captured = std::move(result);
            done = true;
        }
        cv.notify_one();
    });
    ASSERT_TRUE(submit.ok()) << submit.message();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(10), [&] { return done; }));
    }

    ASSERT_TRUE(captured.ok()) << captured.status().message();
    EXPECT_EQ(captured.value().document_id, imported.value().document_id);
    EXPECT_EQ(captured.value().result["fileName"], "托管课程.docx");
    EXPECT_EQ(captured.value().result["fileType"], "docx");

    DocumentMetadataRepository metadata(repository);
    auto loaded = metadata.GetByDocumentId(imported.value().document_id);
    ASSERT_TRUE(loaded.ok()) << loaded.status().message();
    EXPECT_EQ(loaded.value().analysis_status, "completed");
    EXPECT_EQ(loaded.value().analysis_trace_id, "trace-managed-document");

    repository->Close();
    compute.Shutdown(true);
    io.Shutdown(true);
    std::filesystem::remove(source_path, ec);
    std::filesystem::remove_all(store_root, ec);
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path.string() + "-wal", ec);
    std::filesystem::remove(db_path.string() + "-shm", ec);
}

TEST(DocumentAnalysisServiceTest, RetentionCleanupEvictsExpiredManagedDocuments) {
    const auto old_source = TempPath("agent_document_lru_old.docx");
    const auto stale_source = TempPath("agent_document_lru_stale.docx");
    const auto fresh_source = TempPath("agent_document_lru_fresh.docx");
    const auto db_path = TempPath("agent_document_lru.sqlite");
    const auto store_root = TempPath("agent_document_lru_store");
    WriteZip(old_source, {{"word/document.xml", DocxXmlWithParagraph("过期课程文件")}});
    WriteZip(stale_source, {{"word/document.xml", DocxXmlWithParagraph("竞态保护课程文件")}});
    WriteZip(fresh_source, {{"word/document.xml", DocxXmlWithParagraph("近期课程文件")}});
    std::error_code ec;
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove_all(store_root, ec);

    core::ThreadPool compute({1, 8, "document-compute"});
    core::ThreadPool io({1, 8, "document-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    storage::sqlite::SqliteConnectionPoolOptions pool_options;
    pool_options.path = db_path.string();
    pool_options.read_connection_count = 1;
    pool_options.write_connection_count = 1;
    pool_options.busy_timeout_ms = 250;
    auto repository_pool = std::make_shared<storage::sqlite::SqliteConnectionPool>(pool_options);
    ASSERT_TRUE(repository_pool->Start().ok());

    auto file_store = std::make_shared<DocumentFileStore>(
        DocumentFileStoreOptions{store_root, std::chrono::hours(1)});
    DocumentAnalysisService service(compute, io);
    service.SetRetentionCleanupOptions(std::chrono::hours(1), std::chrono::hours(24));
    ASSERT_TRUE(service.SetRepository(repository_pool).ok());
    ASSERT_TRUE(service.SetFileStore(file_store).ok());

    auto old_doc = service.ImportManagedFile(old_source, "过期课程.docx");
    auto stale_doc = service.ImportManagedFile(stale_source, "竞态保护课程.docx");
    auto fresh_doc = service.ImportManagedFile(fresh_source, "近期课程.docx");
    ASSERT_TRUE(old_doc.ok()) << old_doc.status().message();
    ASSERT_TRUE(stale_doc.ok()) << stale_doc.status().message();
    ASSERT_TRUE(fresh_doc.ok()) << fresh_doc.status().message();

    const auto base_ms = 1'700'000'000'000LL;
    const auto cleanup_now_ms = base_ms + std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::hours(2)).count();
    ASSERT_TRUE(service.TouchDocumentAccess(old_doc.value().document_id, base_ms).ok());
    ASSERT_TRUE(service.TouchDocumentAccess(stale_doc.value().document_id, base_ms).ok());
    ASSERT_TRUE(service.TouchDocumentAccess(fresh_doc.value().document_id, cleanup_now_ms).ok());

    DocumentMetadataRepository metadata(repository_pool);
    ASSERT_TRUE(metadata.TouchAccessed(stale_doc.value().document_id, cleanup_now_ms).ok());

    auto cleanup = service.RunRetentionCleanupOnceForTest(cleanup_now_ms);
    ASSERT_TRUE(cleanup.ok()) << cleanup.message();

    EXPECT_FALSE(std::filesystem::exists(store_root / old_doc.value().storage_path));
    EXPECT_TRUE(std::filesystem::exists(store_root / stale_doc.value().storage_path));
    EXPECT_TRUE(std::filesystem::exists(store_root / fresh_doc.value().storage_path));
    EXPECT_EQ(metadata.GetByDocumentId(old_doc.value().document_id).status().code(), core::ErrorCode::NotFound);
    EXPECT_TRUE(metadata.GetByDocumentId(stale_doc.value().document_id).ok());
    EXPECT_TRUE(metadata.GetByDocumentId(fresh_doc.value().document_id).ok());

    ASSERT_TRUE(service.SetFileStore(nullptr).ok());
    ASSERT_TRUE(service.SetRepository(nullptr).ok());
    repository_pool->Close();
    compute.Shutdown(true);
    io.Shutdown(true);
    std::filesystem::remove(old_source, ec);
    std::filesystem::remove(stale_source, ec);
    std::filesystem::remove(fresh_source, ec);
    std::filesystem::remove_all(store_root, ec);
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(db_path.string() + "-wal", ec);
    std::filesystem::remove(db_path.string() + "-shm", ec);
}
