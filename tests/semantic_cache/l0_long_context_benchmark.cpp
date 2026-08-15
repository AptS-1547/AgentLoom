#include "../../src/semantic_cache/l0_memory_cache_adapter.h"
#include "../../src/semantic_cache/redis_connection_pool.h"
#include "../../src/storage/sqlite/sqlite_connection.h"
#include "../../src/vector/embedding_pipeline.h"
#include "../../src/vector/hf_tokenizer.h"
#include "../../src/vector/onnx_text_embedding_model.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>
#include <windows.h>
#include <psapi.h>

namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path dataset_path =
        "D:/Users/21405/source/repos/MyNeuroLikeSystem/data_set/neuro_danmakus_balanced.json";
    std::filesystem::path tokenizer_path =
        "D:/Users/21405/source/repos/AgentBackendPredict/onnx_models/minilm/tokenizer.json";
    std::filesystem::path model_path =
        "D:/Users/21405/source/repos/AgentBackendPredict/onnx_models/minilm/model.onnx";
    std::filesystem::path sqlite_path =
        "D:/Users/21405/source/repos/AgentBackendPredict/data/l0_long_context_benchmark/l0_memory.db";
    std::string redis_host = "127.0.0.1";
    std::string redis_port = "5000";
    std::string user_uuid = "l0-long-context-benchmark";
    std::string execution_provider = "auto";
    std::size_t sample_count = 200;
    std::size_t query_count = 50;
    std::size_t min_text_chars = 8;
    std::size_t max_cached_records = 100;
    std::size_t top_k = 5;
    std::size_t neighbors_per_hit = 2;
    std::size_t print_samples = 0;
    float similarity_floor = 0.35f;
};

struct DialogueRecord {
    std::string user;
    std::string assistant;
};

struct TimedStats {
    std::string name;
    std::vector<double> values_ms;
};

struct MemorySnapshot {
    std::size_t working_set_mb = 0;
    std::size_t private_bytes_mb = 0;
    std::size_t peak_working_set_mb = 0;
};

MemorySnapshot GetMemoryUsage() {
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc));
    return {
        pmc.WorkingSetSize / (1024 * 1024),
        pmc.PrivateUsage / (1024 * 1024),
        pmc.PeakWorkingSetSize / (1024 * 1024),
    };
}

void PrintMemory(std::string_view phase) {
    const auto mem = GetMemoryUsage();
    std::cout << "[memory] " << phase
              << " workingSetMb=" << mem.working_set_mb
              << " privateMb=" << mem.private_bytes_mb
              << " peakMb=" << mem.peak_working_set_mb << "\n";
}

std::string MakeResponse(std::string_view text, std::size_t index);

double ElapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void AddArgValue(int& i, int argc, char** argv, std::string_view name, auto& out) {
    if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + std::string(name));
    }
    std::string value = argv[++i];
    if constexpr (std::is_same_v<std::decay_t<decltype(out)>, std::filesystem::path>) {
        out = value;
    } else if constexpr (std::is_same_v<std::decay_t<decltype(out)>, std::string>) {
        out = value;
    } else if constexpr (std::is_same_v<std::decay_t<decltype(out)>, float>) {
        out = std::stof(value);
    } else {
        out = static_cast<std::decay_t<decltype(out)>>(std::stoull(value));
    }
}

Options ParseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dataset") AddArgValue(i, argc, argv, arg, options.dataset_path);
        else if (arg == "--samples") AddArgValue(i, argc, argv, arg, options.sample_count);
        else if (arg == "--queries") AddArgValue(i, argc, argv, arg, options.query_count);
        else if (arg == "--tokenizer") AddArgValue(i, argc, argv, arg, options.tokenizer_path);
        else if (arg == "--model") AddArgValue(i, argc, argv, arg, options.model_path);
        else if (arg == "--sqlite") AddArgValue(i, argc, argv, arg, options.sqlite_path);
        else if (arg == "--redis-host") AddArgValue(i, argc, argv, arg, options.redis_host);
        else if (arg == "--redis-port") AddArgValue(i, argc, argv, arg, options.redis_port);
        else if (arg == "--user") AddArgValue(i, argc, argv, arg, options.user_uuid);
        else if (arg == "--provider") AddArgValue(i, argc, argv, arg, options.execution_provider);
        else if (arg == "--min-text-chars") AddArgValue(i, argc, argv, arg, options.min_text_chars);
        else if (arg == "--max-cached-records") AddArgValue(i, argc, argv, arg, options.max_cached_records);
        else if (arg == "--top-k") AddArgValue(i, argc, argv, arg, options.top_k);
        else if (arg == "--neighbors") AddArgValue(i, argc, argv, arg, options.neighbors_per_hit);
        else if (arg == "--print-samples") AddArgValue(i, argc, argv, arg, options.print_samples);
        else if (arg == "--similarity-floor") AddArgValue(i, argc, argv, arg, options.similarity_floor);
        else if (arg == "--help") {
            std::cout
                << "Usage: l0_long_context_benchmark [options]\n"
                << "  --dataset PATH\n"
                << "  --samples N\n"
                << "  --queries N\n"
                << "  --tokenizer PATH\n"
                << "  --model PATH\n"
                << "  --sqlite PATH\n"
                << "  --redis-host HOST\n"
                << "  --redis-port PORT\n"
                << "  --provider auto|cpu|cuda|dml\n"
                << "  --top-k N\n"
                << "  --neighbors N\n"
                << "  --print-samples N\n"
                << "  --similarity-floor F\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

std::vector<DialogueRecord> LoadDatasetRecords(const Options& options) {
    std::ifstream file(options.dataset_path);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open dataset: " + options.dataset_path.string());
    }

    Json data = Json::parse(file);
    if (!data.is_array()) {
        throw std::runtime_error("dataset must be a JSON array");
    }

    std::vector<DialogueRecord> records;
    records.reserve(options.sample_count + options.query_count);
    for (const auto& item : data) {
        if (!item.is_object()) {
            continue;
        }
        auto user = item.value("user", std::string{});
        if (user.empty()) {
            user = item.value("text", std::string{});
        }
        auto assistant = item.value("assistant", std::string{});
        if (assistant.empty()) {
            assistant = MakeResponse(user, records.size());
        }
        if (user.size() < options.min_text_chars || assistant.empty()) {
            continue;
        }
        records.push_back({std::move(user), std::move(assistant)});
        if (records.size() >= options.sample_count + options.query_count) {
            break;
        }
    }
    if (records.size() < options.sample_count) {
        throw std::runtime_error("not enough usable dataset rows after filtering");
    }
    return records;
}

void PrintStats(const TimedStats& stats) {
    if (stats.values_ms.empty()) {
        return;
    }
    auto values = stats.values_ms;
    std::sort(values.begin(), values.end());
    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    const auto pct = [&](double q) {
        auto idx = static_cast<std::size_t>(q * static_cast<double>(values.size() - 1));
        return values[idx];
    };
    std::cout << "[stats] " << stats.name
              << " count=" << values.size()
              << " avgMs=" << total / static_cast<double>(values.size())
              << " minMs=" << values.front()
              << " p50Ms=" << pct(0.50)
              << " p95Ms=" << pct(0.95)
              << " p99Ms=" << pct(0.99)
              << " maxMs=" << values.back() << "\n";
}

std::string MakeResponse(std::string_view text, std::size_t index) {
    return "这是一条围绕历史语境的回复 #" + std::to_string(index) +
           "。原始输入是：" + std::string(text);
}

std::string ClipForPrint(std::string value, std::size_t limit) {
    if (value.size() <= limit) {
        return value;
    }
    value.resize(limit);
    value += "...";
    return value;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = ParseArgs(argc, argv);
        std::cout << "=== L0 Long Context Benchmark ===\n";
        std::cout << "[config] dataset=" << options.dataset_path.string() << "\n";
        std::cout << "[config] samples=" << options.sample_count
                  << " queries=" << options.query_count
                  << " topK=" << options.top_k
                  << " neighbors=" << options.neighbors_per_hit
                  << " similarityFloor=" << options.similarity_floor << "\n";
        std::cout << "[config] redis=" << options.redis_host << ":" << options.redis_port
                  << " sqlite=" << options.sqlite_path.string() << "\n";
        PrintMemory("startup");

        auto records = LoadDatasetRecords(options);
        std::cout << "[dataset] usableRecords=" << records.size() << "\n";

        auto tokenizer_result = vector::HfTokenizer::LoadFromFile(options.tokenizer_path);
        if (!tokenizer_result.ok()) {
            std::cerr << "[error] tokenizer: " << tokenizer_result.status().message() << "\n";
            return 1;
        }
        auto tokenizer = std::make_shared<vector::HfTokenizer>(std::move(tokenizer_result.value()));

        vector::EmbeddingModelOptions model_options;
        model_options.model_path = options.model_path;
        model_options.pooling = vector::PoolingStrategy::Mean;
        model_options.normalize = true;
        model_options.execution_provider = options.execution_provider;
        model_options.allow_cpu_fallback = true;
        auto model_result = vector::OnnxTextEmbeddingModel::Load(model_options);
        if (!model_result.ok()) {
            std::cerr << "[error] embedding model: " << model_result.status().message() << "\n";
            return 1;
        }
        auto model = std::shared_ptr<vector::IEmbeddingModel>(std::move(model_result.value()));
        auto embedding = std::make_shared<vector::EmbeddingPipeline>(tokenizer, model);
        std::cout << "[model] provider=" << dynamic_cast<vector::OnnxTextEmbeddingModel*>(model.get())->GetActiveExecutionProvider()
                  << " dim=" << embedding->Dimension() << "\n";
        PrintMemory("after_model_load");

        agent::semantic_cache::RedisPoolOptions redis_options;
        redis_options.host = options.redis_host;
        redis_options.port = options.redis_port;
        redis_options.pool_size = 4;
        auto redis = std::make_shared<agent::semantic_cache::RedisConnectionPool>(redis_options);
        auto redis_status = redis->Start();
        if (!redis_status.ok()) {
            std::cerr << "[error] redis: " << redis_status.message() << "\n";
            return 1;
        }

        std::filesystem::create_directories(options.sqlite_path.parent_path());
        std::error_code ec;
        std::filesystem::remove(options.sqlite_path, ec);
        std::filesystem::remove(options.sqlite_path.string() + "-wal", ec);
        std::filesystem::remove(options.sqlite_path.string() + "-shm", ec);
        storage::sqlite::SqliteConnectionPoolOptions sqlite_options;
        sqlite_options.path = options.sqlite_path.string();
        sqlite_options.read_connection_count = 4;
        sqlite_options.write_connection_count = 1;
        sqlite_options.busy_timeout_ms = 5000;
        sqlite_options.enable_wal = true;
        auto sqlite_pool = std::make_shared<storage::sqlite::SqliteConnectionPool>(std::move(sqlite_options));
        auto sqlite_status = sqlite_pool->Start();
        if (!sqlite_status.ok()) {
            std::cerr << "[error] sqlite: " << sqlite_status.message() << "\n";
            return 1;
        }

        auto index = std::make_shared<agent::semantic_cache::cache_vector::VectorIndexManager>(
            options.user_uuid,
            redis,
            std::move(sqlite_pool),
            options.max_cached_records);
        agent::semantic_cache::L0MemoryCacheAdapterOptions l0_options;
        l0_options.top_k = options.top_k;
        l0_options.neighbors_per_hit = options.neighbors_per_hit;
        l0_options.similarity_floor = options.similarity_floor;
        agent::semantic_cache::L0MemoryCacheAdapter l0(embedding, index, l0_options);

        TimedStats store_stats{"store"};
        for (std::size_t i = 0; i < options.sample_count; ++i) {
            agent::semantic_cache::CacheStoreRequest req;
            req.origin.text = records[i].user;
            req.origin.user_id = options.user_uuid;
            req.origin.session_id = "long-context-session";
            req.response_payload = records[i].assistant;
            const auto start = Clock::now();
            auto status = l0.Store(req);
            store_stats.values_ms.push_back(ElapsedMs(start));
            if (!status.ok()) {
                std::cerr << "[error] store index=" << i << " reason=" << status.message() << "\n";
                return 1;
            }
        }
        PrintStats(store_stats);
        std::cout << "[index] currentSize=" << index->CurrentSize() << "\n";
        PrintMemory("after_store");

        TimedStats lookup_stats{"lookup"};
        std::size_t hits = 0;
        std::size_t payload_bytes = 0;
        double best_score_sum = 0.0;
        const std::size_t query_base = (std::min)(options.sample_count, records.size() - 1);
        const std::size_t query_count = (std::min)(options.query_count, records.size() - query_base);
        for (std::size_t i = 0; i < query_count; ++i) {
            agent::semantic_cache::CacheLookupRequest req;
            req.text = records[query_base + i].user;
            req.user_id = options.user_uuid;
            req.session_id = "long-context-session";
            req.persona_id = "persona-main";
            const auto start = Clock::now();
            auto result = l0.Lookup(req);
            lookup_stats.values_ms.push_back(ElapsedMs(start));
            if (!result.ok()) {
                std::cerr << "[error] lookup index=" << i << " reason=" << result.status().message() << "\n";
                return 1;
            }
            if (result.value().hit) {
                ++hits;
                payload_bytes += result.value().payload.size();
                best_score_sum += result.value().similarity_score;
            }
            if (i < options.print_samples) {
                std::cout << "\n[sample " << i << "] query=" << records[query_base + i].user << "\n"
                          << "[sample " << i << "] hit=" << result.value().hit
                          << " bestScore=" << result.value().similarity_score
                          << " payloadBytes=" << result.value().payload.size() << "\n"
                          << ClipForPrint(result.value().payload, 1200) << "\n";
            }
        }
        PrintStats(lookup_stats);
        std::cout << "[lookup] queries=" << query_count
                  << " hits=" << hits
                  << " hitRate=" << (query_count == 0 ? 0.0 : static_cast<double>(hits) / query_count)
                  << " avgPayloadBytes=" << (hits == 0 ? 0 : payload_bytes / hits)
                  << " avgBestScore=" << (hits == 0 ? 0.0 : best_score_sum / static_cast<double>(hits))
                  << "\n";
        PrintMemory("final");
        redis->Shutdown();
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "[fatal] " << exc.what() << "\n";
        return 1;
    }
}
