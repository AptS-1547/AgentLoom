#include "beast_http_client.h"
#include "document_analysis_service.h"
#include "document_llm_chunk_cache.h"
#include "embedding_pipeline.h"
#include "hf_tokenizer.h"
#include "http_server.h"
#include "isemantic_cache.h"
#include "onnx_text_embedding_model.h"
#include "openai_llm_client.h"
#include "persona_gateway_http_adapter.h"
#include "persona_gateway_service.h"
#include "redis_connection_pool.h"
#include "persona_runtime.h"
#include "session_manager.h"
#include "tls_context.h"
#include "tls_options.h"

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

using Json = nlohmann::json;
namespace asio = boost::asio;
namespace beast = boost::beast;
namespace fs = std::filesystem;
using tcp = asio::ip::tcp;

std::string PathUtf8(const fs::path& path) {
    const auto value = path.u8string();
    return std::string(value.begin(), value.end());
}

std::string ReadFirstLine(const fs::path& path) {
    std::ifstream file(path);
    std::string line;
    std::getline(file, line);
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
        line.pop_back();
    }
    return line;
}

fs::path ResolvePath(const fs::path& config_path, std::string value) {
    if (value.empty()) {
        return {};
    }
    fs::path path(std::move(value));
    if (path.is_absolute()) {
        return path;
    }
    return config_path.parent_path() / path;
}

std::string EnvString(const std::string& name) {
    const char* raw = std::getenv(name.c_str());
    return raw ? std::string(raw) : std::string{};
}

std::string ResolveApiKey(const fs::path& config_path, const Json& llm) {
    const auto env_name = llm.value("api_key_env", std::string{"AGENT_LLM_API_KEY"});
    if (auto key = EnvString(env_name); !key.empty()) {
        return key;
    }
    const auto key_file = llm.value("api_key_file", std::string{});
    if (!key_file.empty()) {
        return ReadFirstLine(ResolvePath(config_path, key_file));
    }
    return {};
}

class LlmClientWithTransport final : public agent::llm::ILlmClient {
public:
    LlmClientWithTransport(std::shared_ptr<agent::net::IHttpClient> transport,
                           std::unique_ptr<agent::llm::OpenAiLlmClient> client)
        : transport_(std::move(transport)),
          client_(std::move(client)) {}

    core::Result<agent::llm::ChatCompletionResponse> Complete(
        const agent::llm::ChatCompletionRequest& req) override {
        return client_->Complete(req);
    }

private:
    std::shared_ptr<agent::net::IHttpClient> transport_;
    std::unique_ptr<agent::llm::OpenAiLlmClient> client_;
};

core::Result<std::shared_ptr<agent::llm::ILlmClient>> CreateLlmClient(
    const fs::path& config_path,
    const Json& root) {
    const auto llm = root.value("llm", Json::object());
    if (!llm.value("enabled", true)) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "llm.enabled is false");
    }

    agent::llm::OpenAiLlmClientOptions options;
    options.base_url = llm.value("base_url", std::string{});
    options.api_key = ResolveApiKey(config_path, llm);
    options.default_model = llm.value("model", std::string{"deepseek-chat"});
    options.timeout_ms = llm.value("timeout_ms", 30000);
    options.retry_policy.max_retries = llm.value("max_retries", 1);
    options.require_api_key = llm.value("require_api_key", true);
    if (options.base_url.empty()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "llm.base_url is empty");
    }
    if (options.require_api_key && options.api_key.empty()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "llm api key was not resolved");
    }

    agent::net::TlsClientOptions tls_options;
#ifdef _WIN32
    if (llm.value("disable_tls_verify_on_windows", true)) {
        tls_options.verify_mode = agent::net::TlsVerifyMode::None;
    }
#endif
    auto tls = agent::net::TlsContext::CreateClient(tls_options);
    if (!tls.ok()) {
        return tls.status();
    }
    agent::net::BeastHttpClientOptions http_options;
    http_options.tls_context = std::move(tls).value();
    auto http = agent::net::BeastHttpClient::Create(std::move(http_options));
    if (!http.ok()) {
        return http.status();
    }
    auto transport = std::shared_ptr<agent::net::IHttpClient>(std::move(http).value());
    auto client = agent::llm::OpenAiLlmClient::Create(std::move(options), *transport);
    if (!client.ok()) {
        return client.status();
    }
    return std::shared_ptr<agent::llm::ILlmClient>(
        std::make_shared<LlmClientWithTransport>(std::move(transport), std::move(client).value()));
}

class DocumentEmbeddingProvider final : public agent::document::IDocumentEmbeddingProvider {
public:
    explicit DocumentEmbeddingProvider(std::shared_ptr<::vector::EmbeddingPipeline> pipeline)
        : pipeline_(std::move(pipeline)) {}

    core::Result<std::vector<float>> EmbedText(std::string_view text) override {
        return pipeline_->Encode(text);
    }

private:
    std::shared_ptr<::vector::EmbeddingPipeline> pipeline_;
};

core::Result<std::shared_ptr<agent::document::IDocumentEmbeddingProvider>> CreateEmbeddingProvider(
    const fs::path& config_path,
    const Json& root) {
    const auto embedding = root.value("embedding", Json::object());
    const auto tokenizer_path = ResolvePath(config_path, embedding.value("tokenizer_path", std::string{"../onnx_models/minilm/tokenizer.json"}));
    const auto model_path = ResolvePath(config_path, embedding.value("model_path", std::string{"../onnx_models/minilm/model.onnx"}));
    if (!fs::exists(tokenizer_path)) {
        return core::Status::Error(core::ErrorCode::NotFound, "tokenizer not found: " + PathUtf8(tokenizer_path));
    }
    if (!fs::exists(model_path)) {
        return core::Status::Error(core::ErrorCode::NotFound, "embedding model not found: " + PathUtf8(model_path));
    }

    auto tokenizer = ::vector::HfTokenizer::LoadFromFile(tokenizer_path);
    if (!tokenizer.ok()) {
        return tokenizer.status();
    }
    auto tokenizer_ptr = std::make_shared<::vector::HfTokenizer>(std::move(tokenizer).value());

    ::vector::EmbeddingModelOptions model_options;
    model_options.model_path = model_path;
    model_options.execution_provider = embedding.value("execution_provider", std::string{"auto"});
    model_options.expected_dimension = static_cast<std::size_t>(embedding.value("dimension", 384));
    model_options.allow_cpu_fallback = true;
    model_options.normalize = true;
    model_options.pooling = ::vector::PoolingStrategy::Mean;
    auto model = ::vector::OnnxTextEmbeddingModel::Load(model_options);
    if (!model.ok()) {
        return model.status();
    }
    std::shared_ptr<::vector::IEmbeddingModel> model_ptr(std::move(model).value());
    auto pipeline = std::make_shared<::vector::EmbeddingPipeline>(std::move(tokenizer_ptr), std::move(model_ptr));
    return std::shared_ptr<agent::document::IDocumentEmbeddingProvider>(
        std::make_shared<DocumentEmbeddingProvider>(std::move(pipeline)));
}

core::Status WarmupEmbeddingProvider(const std::shared_ptr<agent::document::IDocumentEmbeddingProvider>& provider) {
    if (!provider) {
        return core::Status::Ok();
    }
    const auto started = std::chrono::steady_clock::now();
    auto embedded = provider->EmbedText("document analysis embedding warmup");
    if (!embedded.ok()) {
        return embedded.status();
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    std::cout << "[document-e2e] embedding warmup_ms=" << elapsed.count()
              << " dim=" << embedded.value().size() << "\n";
    return core::Status::Ok();
}

std::shared_ptr<agent::document::IDocumentLlmChunkCache> CreateDocumentLlmChunkCache(const Json& root) {
    const auto cache_config = root.value("document_llm_chunk_cache", Json::object());
    if (!cache_config.value("enabled", true)) {
        return nullptr;
    }
    const auto l0 = root.value("l0_memory", Json::object());
    agent::semantic_cache::RedisPoolOptions redis_options;
    redis_options.host = cache_config.value("redis_host", l0.value("redis_host", std::string{"127.0.0.1"}));
    if (cache_config.contains("redis_port")) {
        if (cache_config["redis_port"].is_number_integer()) {
            redis_options.port = std::to_string(cache_config["redis_port"].get<int>());
        } else {
            redis_options.port = cache_config["redis_port"].get<std::string>();
        }
    } else if (l0.contains("redis_port")) {
        if (l0["redis_port"].is_number_integer()) {
            redis_options.port = std::to_string(l0["redis_port"].get<int>());
        } else {
            redis_options.port = l0["redis_port"].get<std::string>();
        }
    } else {
        redis_options.port = "5000";
    }
    redis_options.pool_size = static_cast<std::size_t>(cache_config.value("redis_pool_size", 4));
    auto redis = std::make_shared<agent::semantic_cache::RedisConnectionPool>(redis_options);
    auto status = redis->Start();
    if (!status.ok()) {
        std::cerr << "[document-e2e] document llm cache disabled: " << status.message() << "\n";
        return nullptr;
    }

    agent::document::RedisDocumentLlmChunkCacheOptions options;
    options.key_prefix = cache_config.value("key_prefix", std::string{"agent:e2e:document:llm_chunk"});
    options.ttl = std::chrono::seconds(cache_config.value("ttl_seconds", 7 * 24 * 60 * 60));
    std::cout << "[document-e2e] document llm cache redis=" << redis_options.host
              << ":" << redis_options.port
              << " prefix=" << options.key_prefix << "\n";
    return std::make_shared<agent::document::RedisDocumentLlmChunkCache>(std::move(redis), std::move(options));
}

class NoopSemanticCache final : public agent::semantic_cache::ISemanticCache {
public:
    core::Result<agent::semantic_cache::CacheLookupResult> Lookup(
        const agent::semantic_cache::CacheLookupRequest&) override {
        agent::semantic_cache::CacheLookupResult result;
        result.hit = false;
        return result;
    }

    core::Status Store(const agent::semantic_cache::CacheStoreRequest&) override {
        return core::Status::Ok();
    }
};

std::string StableDocumentIdForPath(const fs::path& path) {
    const auto normalized_path = path.lexically_normal().u8string();
    const std::string normalized(normalized_path.begin(), normalized_path.end());
    std::ostringstream out;
    out << "doc-e2e-" << std::hex << std::hash<std::string>{}(normalized);
    return out.str();
}

::net::BeastHttpResponse SendJsonRequest(std::uint16_t port, const Json& body) {
    asio::io_context io;
    tcp::resolver resolver(io);
    beast::tcp_stream stream(io);
    const auto endpoints = resolver.resolve("127.0.0.1", std::to_string(port));
    stream.connect(endpoints);

    ::net::BeastHttpRequest req{::net::http::verb::post, "/api/document/analyze", 11};
    req.set(::net::http::field::host, "127.0.0.1");
    req.set(::net::http::field::content_type, "application/json");
    req.set("X-Trace-Id", body.value("traceId", "trace-document-e2e"));
    req.body() = body.dump();
    req.prepare_payload();
    ::net::http::write(stream, req);

    beast::flat_buffer buffer;
    ::net::BeastHttpResponse response;
    ::net::http::read(stream, buffer, response);
    beast::error_code ec;
    stream.socket().shutdown(tcp::socket::shutdown_both, ec);
    stream.socket().close(ec);
    return response;
}

std::string FirstTextSample(const Json& chunks, std::size_t max_bytes) {
    std::string out;
    if (!chunks.is_array()) {
        return out;
    }
    for (const auto& chunk : chunks) {
        out.append(chunk.value("title", std::string{})).append("\n");
        out.append(chunk.value("summary", std::string{})).append("\n");
        if (out.size() >= max_bytes) {
            out.resize(max_bytes);
            break;
        }
    }
    return out;
}

std::string TextSample(const Json& items, std::string_view key, std::size_t max_bytes) {
    std::string out;
    if (!items.is_array()) {
        return out;
    }
    for (const auto& item : items) {
        if (!item.is_object()) {
            continue;
        }
        auto it = item.find(std::string(key));
        if (it == item.end() || !it->is_string()) {
            continue;
        }
        if (!out.empty()) {
            out.push_back('\n');
        }
        out.append(it->get<std::string>());
        if (out.size() >= max_bytes) {
            out.resize(max_bytes);
            break;
        }
    }
    return out;
}

std::unordered_set<std::string> ExtractAsciiTerms(std::string_view text) {
    std::unordered_set<std::string> terms;
    std::string current;
    for (const unsigned char ch : text) {
        if (std::isalnum(ch)) {
            current.push_back(static_cast<char>(std::tolower(ch)));
            continue;
        }
        if (current.size() >= 3) {
            terms.insert(current);
        }
        current.clear();
    }
    if (current.size() >= 3) {
        terms.insert(current);
    }
    return terms;
}

double AsciiTermRecall(std::string_view source, std::string_view generated) {
    const auto source_terms = ExtractAsciiTerms(source);
    if (source_terms.empty()) {
        return 1.0;
    }
    const auto generated_terms = ExtractAsciiTerms(generated);
    std::size_t hits = 0;
    for (const auto& term : source_terms) {
        if (generated_terms.contains(term)) {
            ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(source_terms.size());
}

std::unordered_set<std::string> ExtractUtf8NonAsciiUnits(std::string_view text) {
    std::unordered_set<std::string> units;
    for (std::size_t i = 0; i < text.size();) {
        const auto ch = static_cast<unsigned char>(text[i]);
        if (ch < 0x80) {
            ++i;
            continue;
        }
        std::size_t len = 0;
        if ((ch & 0xE0) == 0xC0) {
            len = 2;
        } else if ((ch & 0xF0) == 0xE0) {
            len = 3;
        } else if ((ch & 0xF8) == 0xF0) {
            len = 4;
        } else {
            ++i;
            continue;
        }
        if (i + len > text.size()) {
            break;
        }
        bool valid = true;
        for (std::size_t j = 1; j < len; ++j) {
            if ((static_cast<unsigned char>(text[i + j]) & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }
        if (valid) {
            units.insert(std::string(text.substr(i, len)));
            i += len;
            continue;
        }
        ++i;
    }
    return units;
}

double Utf8NonAsciiUnitRecall(std::string_view source, std::string_view generated) {
    const auto source_units = ExtractUtf8NonAsciiUnits(source);
    if (source_units.empty()) {
        return 1.0;
    }
    const auto generated_units = ExtractUtf8NonAsciiUnits(generated);
    std::size_t hits = 0;
    for (const auto& unit : source_units) {
        if (generated_units.contains(unit)) {
            ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(source_units.size());
}

std::size_t CountChunksBySource(const Json& chunks, std::string_view source) {
    if (!chunks.is_array()) {
        return 0;
    }
    std::size_t count = 0;
    for (const auto& chunk : chunks) {
        if (chunk.is_object() && chunk.value("source", std::string{}) == source) {
            ++count;
        }
    }
    return count;
}

std::string SafeReportName(std::string value) {
    if (value.empty()) {
        value = "document";
    }
    for (auto& ch : value) {
        const auto c = static_cast<unsigned char>(ch);
        if (!std::isalnum(c) && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    return value;
}

core::Status WriteQualityReport(const fs::path& output_dir, const Json& body) {
    fs::create_directories(output_dir);
    const auto& data = body.at("data");
    const auto path = output_dir / (SafeReportName(body.value("documentId", std::string{})) + ".json");
    const auto source_sample = TextSample(data["blocks"], "text", 6000);
    const auto summary_sample = FirstTextSample(data["chunks"], 6000);
    const auto indexed_text_sample = TextSample(data["chunks"], "text", 6000);
    const auto llm_chunk_count = CountChunksBySource(data["chunks"], "llm");
    Json report{
        {"documentId", body.value("documentId", std::string{})},
        {"fileName", data.value("fileName", std::string{})},
        {"blockCount", data["blocks"].size()},
        {"chunkCount", data["chunks"].size()},
        {"llmChunkCount", llm_chunk_count},
        {"localChunkCount", CountChunksBySource(data["chunks"], "local")},
        {"diagnosisScore", data["diagnosis"].value("score", 0.0)},
        {"tokenCount", data.value("tokenCount", 0)},
        {"summaryAsciiTermRecall", AsciiTermRecall(source_sample, summary_sample)},
        {"summaryUtf8NonAsciiUnitRecall", Utf8NonAsciiUnitRecall(source_sample, summary_sample)},
        {"indexedTextAsciiTermRecall", AsciiTermRecall(source_sample, indexed_text_sample)},
        {"indexedTextUtf8NonAsciiUnitRecall", Utf8NonAsciiUnitRecall(source_sample, indexed_text_sample)},
        {"sourceEqualsIndexedText", source_sample == indexed_text_sample},
        {"runNodes", data["runNodes"]},
        {"sourceSample", source_sample},
        {"summarySample", summary_sample},
        {"indexedTextSample", indexed_text_sample},
        {"chunks", data["chunks"]},
        {"mindmap", data["mindmap"]},
        {"diagnosis", data["diagnosis"]},
    };
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to open report: " + PathUtf8(path));
    }
    out << report.dump(2, ' ', false, Json::error_handler_t::replace);
    std::cout << "[document-e2e] report: " << PathUtf8(path)
              << " llm_chunks=" << llm_chunk_count
              << " summary_ascii_recall=" << report["summaryAsciiTermRecall"].get<double>()
              << " summary_utf8_recall=" << report["summaryUtf8NonAsciiUnitRecall"].get<double>()
              << " indexed_ascii_recall=" << report["indexedTextAsciiTermRecall"].get<double>()
              << " indexed_utf8_recall=" << report["indexedTextUtf8NonAsciiUnitRecall"].get<double>()
              << "\n";
    return core::Status::Ok();
}

int Fail(const std::string& message) {
    std::cerr << "[document-e2e] error: " << message << "\n";
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <tools/persona_gateway_e2e_server.json> <docx-or-pptx> [more docs...]\n";
        return 2;
    }

    try {
        const fs::path config_path = argv[1];
        std::ifstream config_file(config_path);
        if (!config_file) {
            return Fail("cannot open config: " + PathUtf8(config_path));
        }
        const auto config = Json::parse(config_file);

        auto llm = CreateLlmClient(config_path, config);
        if (!llm.ok()) {
            return Fail("llm client: " + llm.status().message());
        }
        auto embedding = CreateEmbeddingProvider(config_path, config);
        if (!embedding.ok()) {
            return Fail("embedding provider: " + embedding.status().message());
        }
        if (auto status = WarmupEmbeddingProvider(embedding.value()); !status.ok()) {
            return Fail("embedding warmup: " + status.message());
        }
        auto llm_chunk_cache = CreateDocumentLlmChunkCache(config);

        core::ThreadPool compute({2, 128, "document-e2e-compute"});
        core::ThreadPool io({4, 128, "document-e2e-io"});
        if (auto status = compute.Start(); !status.ok()) {
            return Fail("compute pool: " + status.message());
        }
        if (auto status = io.Start(); !status.ok()) {
            return Fail("io pool: " + status.message());
        }

        agent::service::persona::SessionManager sessions(compute, io);
        auto memory = std::make_shared<agent::service::persona::SemanticMemoryContextProvider>(
            std::make_shared<NoopSemanticCache>());
        auto emotion = std::make_shared<agent::service::persona::NeutralEmotionAnalyzer>();
        agent::service::persona::PersonaRuntime runtime(
            sessions,
            memory,
            emotion,
            llm.value(),
            agent::service::persona::PersonaRuntimeOptions{.recent_raw_turns = 10});
        agent::service::gateway::PersonaGatewayService gateway(sessions, runtime, nullptr);
        auto document_service = std::make_shared<agent::document::DocumentAnalysisService>(compute, io);
        agent::service::gateway::PersonaGatewayHttpAdapter adapter(
            gateway,
            nullptr,
            nullptr,
            document_service,
            llm.value(),
            embedding.value(),
            std::move(llm_chunk_cache));

        ::net::HttpServer server({.address = "127.0.0.1", .port = 0, .io_threads = 1});
        server.SetHttpRequestHandler([&adapter](std::shared_ptr<::net::IHttpRequest> request) {
            adapter.HandleHttp(std::move(request));
        });
        auto start = server.Start();
        if (!start.ok()) {
            return Fail("http server: " + start.message());
        }

        const auto output_dir = fs::current_path() / "test_dumps" / "document_quality";
        const auto chunk_llm_model = config.value("llm", Json::object()).value("model", std::string{});
        for (int i = 2; i < argc; ++i) {
            const fs::path doc_path = argv[i];
            const auto doc_id = StableDocumentIdForPath(doc_path) + "-manual-real";
            Json request{
                {"traceId", "trace-document-manual-e2e"},
                {"documentId", doc_id},
                {"path", PathUtf8(doc_path)},
                {"fileName", PathUtf8(doc_path.filename())},
                {"maxChunkSlices", 5},
                {"enableEmbeddingClustering", true},
                {"enableLlmChunkFallback", true},
                {"chunkLlmModel", chunk_llm_model},
                {"chunkLlmMaxTokens", 900},
            };
            std::cout << "[document-e2e] analyzing: " << PathUtf8(doc_path) << "\n";
            auto response = SendJsonRequest(server.port(), request);
            if (response.result() != ::net::http::status::ok) {
                server.Stop();
                return Fail("HTTP " + std::to_string(response.result_int()) + ": " + response.body());
            }
            auto body = Json::parse(response.body());
            if (!body.value("ok", false)) {
                server.Stop();
                return Fail("response ok=false: " + response.body());
            }
            const auto& data = body["data"];
            std::cout << "[document-e2e] chunks=" << data["chunks"].size()
                      << " blocks=" << data["blocks"].size()
                      << " latency_ms=" << body.value("latencyMs", 0)
                      << "\n";
            if (auto status = WriteQualityReport(output_dir, body); !status.ok()) {
                server.Stop();
                return Fail(status.message());
            }
        }

        server.Stop();
        compute.Shutdown(true);
        io.Shutdown(true);
        return 0;
    } catch (const std::exception& e) {
        return Fail(e.what());
    }
}
