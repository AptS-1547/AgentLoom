// 显式 opt-in 的真实 LLM Skill 工具调用 E2E；不纳入默认 CTest，避免默认测试访问外部 API。
//
// 多工具判别版：注册 4 类下游典型工具（视觉观察 / 文档分析 / 知识库查询 / 客户画像），
// 每个工具写入一条 L4 种子记忆，用「正样本 + 近邻负样本 + 无关负样本」矩阵测量
// 向量召回的命中率、top-1 正确率与分数区分度，并做一轮 LLM 工具路由验证。
#include "option_parser.h"
#include "server_options.h"
#include "openai_llm_client.h"
#include "beast_http_client.h"
#include "tls_context.h"
#include "skill_manifest.h"
#include "skill_prompt_compiler.h"
#include "skill_executor.h"
#include "tool_memory_provider.h"
#include "sqlite_vector_repository.h"
#include "sqlite_connection_pool.h"
#include "vector_partition_registry.h"
#include "vector_index_manager.h"
#include "embedding_pipeline.h"
#include "hf_tokenizer.h"
#include "onnx_text_embedding_model.h"

#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>
#include <unordered_map>

namespace {
using Json = nlohmann::json;

// 判别阈值：正样本期望目标工具分数显著高于该值，无关负样本期望所有分数低于该值。
constexpr double kDiscrimThreshold = 0.5;

struct Case {
    std::string id;
    std::string prompt;
    std::string expect_tool_id;  // 空串 = 期望不召回/不调用任何工具
};

// 一个工具的完整定义 + 其 L4 种子记忆。
struct ToolSeed {
    std::string skill_id;
    std::string tool_name;
    std::string description;
    std::string prompt_instruction;
    std::string input_schema_json;
    std::string l4_payload;      // 种子语义描述（用于 embedding 与 LLM 上下文）
    std::string l4_instruction;  // 种子工具指令
    std::string l4_schema;       // 种子 schema
    std::string memory_hash;     // global:l4:<skill_id>:<intent>
    std::vector<std::string> keywords;  // 触发关键词，编译为确定性正则兜底
    std::vector<std::string> negative_keywords;  // 否定排除词，命中即取消正则触发
};

// 通用录制型执行器：按工具自身回填一个可辨识的假结果，不依赖具体工具。
class RecordingNativeExecutor final : public agent::skill::ISkillExecutor {
public:
    core::Result<agent::service::persona::SkillSessionSnapshot> Start(
        const agent::skill::SkillExecutionRequest& request,
        agent::skill::SkillExecutionCallbacks callbacks) override {
        if (callbacks.on_result) {
            agent::skill::SkillResult result;
            result.call_id = request.call.call_id;
            result.skill_id = request.call.skill_id;
            result.result_json = R"({"source":"e2e_fake_backend","tool":")" + request.call.skill_id + R"(","ok":true})";
            callbacks.on_result(std::move(result));
        }
        return agent::service::persona::SkillSessionSnapshot{};
    }
    core::Status Cancel(std::string_view) override { return core::Status::Ok(); }
};

int Fail(std::string message) { std::cerr << "[skill-e2e] FAIL: " << message << '\n'; return 1; }

// 下游典型的四类工具；skill_id 带点号，tool_name 为 OpenAI-compatible 函数名。
std::vector<ToolSeed> BuildToolSeeds() {
    return {
        {"vision.observe", "vision_observe",
         "Observe the current visual input (screen, camera, or visible content).",
         "Call only when visual observation is requested; never invent an observation before the tool result.",
         R"({"type":"object","required":["reason"],"properties":{"reason":{"type":"string"}}})",
         "用户请求观察画面、摄像头、屏幕或当前可见内容时使用 vision.observe 视觉观察工具。",
         "需要外部视觉观察时调用 vision.observe。没有工具结果前不要编造画面内容。",
         R"({"type":"object","required":["reason"],"properties":{"reason":{"type":"string"}}})",
         "global:l4:vision.observe:intent.camera",
         {"屏幕", "画面", "摄像头", "screen", "camera", "visible"},
         {"不用看", "别看", "不要看", "无需看"}},
        {"document.analyze", "document_analyze",
         "Analyze a document's content (contract clauses, terms, tables, or text).",
         "Call when the user asks to read, analyze, or summarize a document, file, contract, or table.",
         R"({"type":"object","required":["doc_id"],"properties":{"doc_id":{"type":"string"},"question":{"type":"string"}}})",
         "用户请求分析文档、合同、文件、条款、表格或文本内容时使用 document.analyze 文档分析工具。",
         "需要阅读或分析文档/合同/文件内容时调用 document.analyze。不要在没有文档结果前编造条款内容。",
         R"({"type":"object","required":["doc_id"],"properties":{"doc_id":{"type":"string"},"question":{"type":"string"}}})",
         "global:l4:document.analyze:intent.doc",
         {"文档", "合同", "条款", "文件", "document", "contract", "clause"},
         {}},
        {"kb.query", "kb_query",
         "Retrieve sales knowledge, talk tracks, product facts, or objection-handling guidance.",
         "Call when the user asks for sales knowledge, talk tracks, objection handling, or product facts.",
         R"({"type":"object","required":["query"],"properties":{"query":{"type":"string"}}})",
         "用户请求销售话术、产品知识、异议处理、谈判技巧等销售知识时使用 kb.query 知识库查询工具。",
         "需要检索销售知识、话术或产品事实时调用 kb.query。不要在没有知识库结果前编造销售建议。",
         R"({"type":"object","required":["query"],"properties":{"query":{"type":"string"}}})",
         "global:l4:kb.query:intent.sales_knowledge",
         {"话术", "异议", "压价", "谈判", "产品知识", "objection"},
         {}},
        {"profile.read", "profile_read",
         "Read a customer's profile, preferences, purchase history, or interaction history.",
         "Call when the user asks about a customer's profile, preferences, history, or persona.",
         R"({"type":"object","required":["customer_id"],"properties":{"customer_id":{"type":"string"}}})",
         "用户请求读取客户画像、偏好、购买记录或交互历史时使用 profile.read 客户画像工具。",
         "需要读取客户画像、偏好或历史时调用 profile.read。不要在没有画像结果前编造客户信息。",
         R"({"type":"object","required":["customer_id"],"properties":{"customer_id":{"type":"string"}}})",
         "global:l4:profile.read:intent.customer",
         {"画像", "偏好", "购买记录", "历史订单", "客户资料", "preference"},
         {}},
    };
}

// 判别用例矩阵：正样本、近邻负样本（含他类关键词）、无关负样本。
std::vector<Case> BuildCases() {
    return {
        {"vision_en", "Please inspect what is currently visible on my screen.", "vision.observe"},
        {"vision_cn", "帮我看看屏幕上的表格内容。", "vision.observe"},
        {"doc_en", "Analyze the liability clauses in this contract document.", "document.analyze"},
        {"doc_cn", "分析这份合同里的违约责任条款。", "document.analyze"},
        {"kb_en", "How should I handle a customer who keeps pushing for a lower price?", "kb.query"},
        {"kb_cn", "客户一直压价，我应该怎么应对？", "kb.query"},
        {"profile_en", "What are this customer's preferences and purchase history?", "profile.read"},
        {"profile_cn", "这个客户的偏好和过往购买记录是什么？", "profile.read"},
        // 近邻负样本：携带他类关键词，期望仍命中正确工具而非误命中他类。
        {"nearmiss_doc", "查看一下这份合同文档里的条款内容，不用看屏幕。", "document.analyze"},
        {"nearmiss_profile", "读取这个客户的画像和偏好信息。", "profile.read"},
        {"nearmiss_kb", "检索一下销售话术库来应对客户的异议。", "kb.query"},
        // 无关负样本：期望不召回任何工具。
        {"neg_funnel", "用一句话解释什么是销售漏斗。", ""},
        {"neg_weather", "今天天气怎么样？", ""},
        {"neg_joke", "给我讲个冷笑话。", ""},
    };
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json> [report.jsonl]\n";
        return 2;
    }
    const std::filesystem::path config_path = argv[1];
    const std::filesystem::path report_path = argc > 2 ? argv[2] : "skill_llm_e2e_report.jsonl";
    MultimodalServerOptions options;
    try {
        std::vector<std::string> args{"skill-e2e", "--llm", "skill-e2e.gguf", "--config", config_path.string()};
        std::vector<char*> raw; for (auto& arg : args) raw.push_back(arg.data());
        options = ParseMultimodalOptions(static_cast<int>(raw.size()), raw.data());
    } catch (const std::exception& error) { return Fail(std::string("config parse: ") + error.what()); }
    if (options.llm.base_url.empty() || options.llm.api_key.empty()) return Fail("llm endpoint or API key is missing");

    agent::net::TlsClientOptions tls_options;
#ifdef _WIN32
    std::filesystem::path ca = options.llm.ca_bundle_path;
    if (ca.empty()) {
        ca = config_path.parent_path() / "../config/certs/mozilla-ca-bundle.pem";
    } else if (ca.is_relative()) {
        ca = options.config_file_path.parent_path() / ca;
    }
    if (!std::filesystem::exists(ca)) return Fail("Windows CA bundle not found: " + ca.string());
    tls_options.ca_bundle_path = ca.string();
#endif
    auto tls = agent::net::TlsContext::CreateClient(tls_options);
    if (!tls) return Fail("TLS: " + tls.status().message());
    auto http = agent::net::BeastHttpClient::Create({.tls_context = std::move(tls).value()});
    if (!http) return Fail("HTTP: " + http.status().message());
    agent::llm::OpenAiLlmClientOptions client_options;
    client_options.base_url = options.llm.base_url;
    client_options.api_key = options.llm.api_key;
    client_options.default_model = options.llm.model;
    client_options.timeout_ms = options.llm.timeout_ms;
    client_options.retry_policy.max_retries = options.llm.max_retries;
    auto client = agent::llm::OpenAiLlmClient::Create(std::move(client_options), *http.value());
    if (!client) return Fail("LLM client: " + client.status().message());

    const auto tools = BuildToolSeeds();
    const auto cases = BuildCases();
    auto registry = std::make_shared<agent::skill::InMemorySkillRegistry>();
    std::unordered_map<std::string, ToolSeed> by_skill_id;
    for (const auto& seed : tools) {
        agent::skill::SkillManifest manifest;
        manifest.skill_id = seed.skill_id;
        manifest.tool_name = seed.tool_name;
        manifest.version = "e2e-1";
        manifest.description = seed.description;
        manifest.prompt_instruction = seed.prompt_instruction;
        manifest.input_schema_json = seed.input_schema_json;
        manifest.executor.reference = seed.skill_id;
        if (auto status = registry->Register(manifest); !status.ok()) return Fail(status.message());
        by_skill_id.emplace(seed.skill_id, seed);
    }

    if (options.embedding.tokenizer_path.empty() || options.embedding.onnx_model_path.empty()) {
        return Fail("embedding configuration is required for L4 E2E");
    }
    auto tokenizer_path = std::filesystem::path(options.embedding.tokenizer_path);
    auto embedding_model_path = std::filesystem::path(options.embedding.onnx_model_path);
    if (tokenizer_path.is_relative()) tokenizer_path = config_path.parent_path() / tokenizer_path;
    if (embedding_model_path.is_relative()) embedding_model_path = config_path.parent_path() / embedding_model_path;
    auto tokenizer = vector::HfTokenizer::LoadFromFile(tokenizer_path);
    if (!tokenizer) return Fail("embedding tokenizer: " + tokenizer.status().message());
    vector::EmbeddingModelOptions model_options;
    model_options.model_path = embedding_model_path.string();
    model_options.execution_provider = options.embedding.execution_provider;
    model_options.allow_cpu_fallback = true;
    model_options.expected_dimension = static_cast<std::size_t>(options.embedding.expected_dimension);
    model_options.pooling = vector::PoolingStrategy::Mean;
    model_options.normalize = true;
    auto embedding_model = vector::OnnxTextEmbeddingModel::Load(model_options);
    if (!embedding_model) return Fail("embedding model: " + embedding_model.status().message());
    auto embedding = std::make_shared<vector::EmbeddingPipeline>(
        std::make_shared<vector::HfTokenizer>(std::move(tokenizer).value()),
        std::shared_ptr<vector::IEmbeddingModel>(std::move(embedding_model).value()));

    const auto l4_db_path = std::filesystem::temp_directory_path() / "agentloom_skill_l4_e2e.sqlite";
    std::error_code cleanup_error;
    std::filesystem::remove(l4_db_path, cleanup_error);
    std::filesystem::remove(l4_db_path.string() + "-wal", cleanup_error);
    std::filesystem::remove(l4_db_path.string() + "-shm", cleanup_error);
    storage::sqlite::SqliteConnectionPoolOptions pool_options;
    pool_options.path = l4_db_path.string();
    pool_options.read_connection_count = 2;
    pool_options.write_connection_count = 1;
    pool_options.enable_wal = true;
    auto sqlite_pool = std::make_shared<storage::sqlite::SqliteConnectionPool>(pool_options);
    if (auto status = sqlite_pool->Start(); !status.ok()) return Fail("L4 sqlite start: " + status.message());
    auto repository = std::make_shared<agent::vector_storage::SqliteVectorRepository>(sqlite_pool);
    if (auto status = repository->EnsureSchema(); !status.ok()) return Fail("L4 schema: " + status.message());
    agent::vector_storage::CollectionDescriptor collection;
    collection.name = "skill_l4";
    collection.embedding_model_fingerprint = "configured";
    collection.tokenizer_fingerprint = "configured";
    collection.pooling_strategy = "mean";
    collection.normalization = "l2";
    collection.dimension = embedding->Dimension();
    collection.corpus_version = "skill-e2e";
    collection.policy_version = "skill-e2e";
    auto collection_id = repository->EnsureCollection(collection);
    if (!collection_id) return Fail("L4 collection: " + collection_id.status().message());
    auto partition_registry = std::make_shared<agent::vector_storage::PartitionRegistry>(repository);
    agent::vector_storage::PartitionKey partition_key;
    partition_key.collection_id = collection_id.value();
    partition_key.memory_level = "L4";
    auto partition = partition_registry->Resolve(partition_key);
    if (!partition) return Fail("L4 partition: " + partition.status().message());

    // 为每个工具写入一条 L4 种子记忆。
    for (const auto& seed : tools) {
        auto l4_vector = embedding->Encode(seed.l4_payload);
        if (!l4_vector) return Fail("L4 embedding " + seed.skill_id + ": " + l4_vector.status().message());
        agent::vector_storage::EntryRecord l4_entry;
        l4_entry.partition_id = partition.value();
        l4_entry.cache_key = seed.tool_name;
        l4_entry.text_hash = "skill-l4-canonical";
        l4_entry.content_hash = "skill-l4-canonical";
        l4_entry.memory_hash = seed.memory_hash;
        l4_entry.vector = l4_vector.value();
        l4_entry.memory_type = "capability";
        l4_entry.emotion_intensity = 0.0F;
        l4_entry.payload = seed.l4_payload;
        Json meta{{"tool_id", seed.skill_id}, {"instruction", seed.l4_instruction},
                  {"schema", seed.l4_schema}, {"priority", 100}};
        l4_entry.extra_metadata_json = meta.dump();
        if (auto inserted = repository->InsertEntry(std::move(l4_entry)); !inserted) return Fail("L4 insert: " + inserted.status().message());
    }

    auto index_manager = std::make_shared<agent::vector::VectorIndexManager>(
        repository, partition_registry, embedding->Dimension(), agent::vector::IndexManagerOptions{});
    agent::service::persona::VectorToolMemoryProviderOptions provider_options;
    provider_options.collection_id = collection_id.value();
    provider_options.top_k = 3;  // 调回生产默认 top_k。
    provider_options.min_score = 0.0;  // 阈值归零以观测完整分数分布，判别由 kDiscrimThreshold 评估。
    provider_options.enable_regex = true;  // 开启工具级关键词正则兜底。
    provider_options.enable_vector = true;
    for (const auto& seed : tools) {
        agent::service::persona::ToolKeywordTrigger trigger;
        trigger.tool_id = seed.skill_id;
        trigger.instruction = seed.l4_instruction;
        trigger.schema_json = seed.l4_schema;
        trigger.keywords = seed.keywords;
        trigger.negative_keywords = seed.negative_keywords;
        provider_options.keyword_triggers.push_back(std::move(trigger));
    }
    agent::service::persona::VectorToolMemoryProvider tool_provider(
        embedding, index_manager, partition_registry, provider_options);
    agent::skill::SkillPromptCompiler compiler(registry);
    auto executor_factory = std::make_shared<agent::skill::InMemorySkillExecutorFactory>();
    if (auto status = executor_factory->Register("native", std::make_shared<RecordingNativeExecutor>()); !status.ok()) return Fail(status.message());
    auto skill_sessions = std::make_shared<agent::service::persona::SkillSessionManager>();
    auto invocation = std::make_shared<agent::skill::SkillInvocationService>(registry, executor_factory, skill_sessions);
    agent::skill::SkillToolCallCoordinator coordinator(registry, invocation);

    std::ofstream report(report_path, std::ios::binary | std::ios::trunc);
    if (!report) return Fail("cannot open report output");

    std::size_t recall_pass = 0;
    std::size_t positive_count = 0;
    std::size_t top1_correct = 0;
    std::size_t false_positive_count = 0;
    std::size_t llm_routed = 0;
    std::size_t llm_routed_ok = 0;

    for (const auto& test : cases) {
        const bool positive = !test.expect_tool_id.empty();
        agent::service::persona::ToolMemoryQuery memory_query;
        memory_query.session_id = "skill-e2e-session";
        memory_query.user_uuid = "skill-e2e-user";
        memory_query.persona_id = "skill-e2e-persona";
        memory_query.trace_id = "skill-e2e-" + test.id;
        memory_query.query = test.prompt;
        auto l4_context = tool_provider.Query(memory_query);
        if (!l4_context) return Fail("L4 query " + test.id + ": " + l4_context.status().message());

        // 收集召回命中，计算判别指标。
        Json hits = Json::array();
        std::string top1_tool;
        double top1_score = 0.0;
        double expect_score = 0.0;
        double max_wrong_score = 0.0;
        bool expect_in_hits = false;
        for (const auto& hit : l4_context.value().hits) {
            hits.push_back({{"tool_id", hit.tool_id}, {"score", hit.score},
                            {"regex_hit", hit.regex_hit}, {"vector_hit", hit.vector_hit}});
            if (top1_tool.empty()) { top1_tool = hit.tool_id; top1_score = hit.score; }
            if (positive && hit.tool_id == test.expect_tool_id) {
                expect_in_hits = true;
                expect_score = hit.score;
            } else if (!positive || hit.tool_id != test.expect_tool_id) {
                max_wrong_score = std::max(max_wrong_score, hit.score);
            }
        }
        const bool top1_correct_case = positive && (top1_tool == test.expect_tool_id);
        const bool false_positive_case = (positive && max_wrong_score >= kDiscrimThreshold) ||
                                         (!positive && !hits.empty() && top1_score >= kDiscrimThreshold);
        const double score_margin = positive
            ? (expect_in_hits ? expect_score - max_wrong_score : -1.0f)
            : -top1_score;
        const bool recall_ok_case = positive
            ? (top1_correct_case && !false_positive_case)
            : !false_positive_case;

        if (recall_ok_case) ++recall_pass;
        if (positive) {
            ++positive_count;
            if (top1_correct_case) ++top1_correct;
        }
        if (false_positive_case) ++false_positive_count;

        Json row{{"case_id", test.id}, {"prompt", test.prompt}, {"expect_tool_id", test.expect_tool_id},
                 {"positive", positive}, {"hits", std::move(hits)},
                 {"top1_tool_id", top1_tool}, {"top1_score", top1_score},
                 {"score_margin", score_margin}, {"false_positive", false_positive_case},
                 {"recall_ok", recall_ok_case}};

        // LLM 工具路由验证：仅正样本；只注入召回的 top-k 工具（静态提示 + 工具定义都一致）。
        if (positive) {
            ++llm_routed;
            std::vector<std::string> recalled_ids;
            for (const auto& hit : l4_context.value().hits) recalled_ids.push_back(hit.tool_id);
            auto prompt = compiler.Compile({recalled_ids, {}, {}});
            if (!prompt) return Fail("prompt compile: " + prompt.status().message());
            agent::llm::ChatCompletionRequest request;
            request.messages.push_back({agent::llm::ChatRole::System,
                                        prompt.value() + "\n" + l4_context.value().prompt_block});
            request.messages.push_back({agent::llm::ChatRole::User, test.prompt});
            for (const auto& hit : l4_context.value().hits) {
                auto it = by_skill_id.find(hit.tool_id);
                if (it == by_skill_id.end()) continue;
                agent::llm::ChatCompletionRequest::Tool tool;
                tool.name = it->second.tool_name;
                tool.description = it->second.description;
                tool.parameters_json = it->second.input_schema_json;
                request.tools.push_back(std::move(tool));
            }
            request.tool_choice = "auto";
            auto response = client.value()->Complete(request);
            const auto latency = std::chrono::milliseconds(0);  // 路由轮不统计精细延迟，保持报告字段一致。
            row["llm_ok"] = response.ok();
            if (response.ok()) {
                bool called_expected = false;
                Json names = Json::array();
                for (const auto& call : response.value().tool_calls) {
                    names.push_back(call.name);
                    auto it = by_skill_id.find(test.expect_tool_id);
                    if (it != by_skill_id.end() && call.name == it->second.tool_name) called_expected = true;
                }
                row["llm_tool_calls"] = std::move(names);
                row["llm_called_expected"] = called_expected;
                if (called_expected) ++llm_routed_ok;
            } else {
                row["llm_error"] = response.status().message();
            }
            row["latency_ms"] = latency.count();
        }

        report << row.dump() << '\n';
        std::cout << "[skill-e2e] " << test.id
                  << " top1=" << (top1_tool.empty() ? "<none>" : top1_tool)
                  << " score=" << top1_score
                  << " recall=" << (recall_ok_case ? "ok" : "FAIL") << '\n';
    }

    std::cout << "[skill-e2e] recall_pass=" << recall_pass << "/" << cases.size()
              << " top1_accuracy=" << (positive_count ? top1_correct : 0) << "/" << positive_count
              << " false_positive=" << false_positive_count
              << " llm_route=" << (llm_routed ? llm_routed_ok : 0) << "/" << llm_routed
              << " report=" << report_path.string() << '\n';
    return recall_pass == cases.size() ? 0 : 1;
}
