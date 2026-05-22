#include "option_parser.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class ArgvBuilder {
public:
    ArgvBuilder(std::initializer_list<std::string_view> args) {
        storage_.reserve(args.size());
        argv_.reserve(args.size());
        for (std::string_view arg : args) {
            storage_.emplace_back(arg);
        }
        for (std::string& arg : storage_) {
            argv_.push_back(arg.data());
        }
    }

    explicit ArgvBuilder(std::vector<std::string> args)
        : storage_(std::move(args)) {
        argv_.reserve(storage_.size());
        for (std::string& arg : storage_) {
            argv_.push_back(arg.data());
        }
    }

    int argc() const noexcept {
        return static_cast<int>(argv_.size());
    }

    char** argv() noexcept {
        return argv_.data();
    }

private:
    std::vector<std::string> storage_;
    std::vector<char*> argv_;
};

class ScopedTempDirectory {
public:
    explicit ScopedTempDirectory(std::string_view prefix)
        : path_(std::filesystem::temp_directory_path() /
                (std::string(prefix) + "_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
                 std::to_string(std::rand()))) {
        std::filesystem::create_directories(path_);
    }

    ~ScopedTempDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

std::filesystem::path WriteFile(const std::filesystem::path& path, std::string_view content) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to create test file");
    }
    file << content;
    return path;
}

MultimodalServerOptions Parse(std::initializer_list<std::string_view> args) {
    ArgvBuilder argv(args);
    return ParseMultimodalOptions(argv.argc(), argv.argv());
}

} // namespace

TEST(ConfigOptionParserTest, DetectsHelpFlag) {
    ArgvBuilder long_help({"server", "--help"});
    EXPECT_TRUE(IsHelpRequested(long_help.argc(), long_help.argv()));

    ArgvBuilder short_help({"server", "-h"});
    EXPECT_TRUE(IsHelpRequested(short_help.argc(), short_help.argv()));

    ArgvBuilder no_help({"server", "--llm", "model.gguf"});
    EXPECT_FALSE(IsHelpRequested(no_help.argc(), no_help.argv()));
}

TEST(ConfigOptionParserTest, ParsesCliFallbackOptionsThroughSections) {
    auto options = Parse({
        "server",
        "--llm", "llm.gguf",
        "--bert", "bert.onnx",
        "--vit", "vit.onnx",
        "--mmproj", "mmproj.gguf",
        "--ngl", "12",
        "--provider", "cuda",
        "--cuda-device", "1",
        "--host", "0.0.0.0",
        "--port", "6000",
        "--grpc-num-cqs", "2",
        "--grpc-min-pollers", "3",
        "--grpc-max-pollers", "2",
        "--max-recv-mb", "128",
        "--max-send-mb", "32",
        "--stats-log-interval-seconds", "-1",
        "--slow-request-ms", "-5",
        "--auth-token", "token-123",
        "--auth-header", "x-test-auth",
        "--max-image-mb", "8",
        "--max-context-size", "2048",
        "--vlm-cache-enabled",
        "--vlm-cache-persist",
        "--vlm-cache-dir", "cache/test",
        "--vram-warning-free-mb", "256",
        "--vram-unload-free-mb", "128"
    });

    EXPECT_EQ(options.llm_model, "llm.gguf");
    EXPECT_EQ(options.bert_model, "bert.onnx");
    EXPECT_EQ(options.vit_model, "vit.onnx");
    EXPECT_EQ(options.mmproj, "mmproj.gguf");
    EXPECT_EQ(options.n_gpu_layers, 12);
    EXPECT_EQ(options.bert_runtime.execution_provider, "cuda");
    EXPECT_EQ(options.bert_runtime.cuda_device_id, 1);
    EXPECT_EQ(options.grpc.host, "0.0.0.0");
    EXPECT_EQ(options.grpc.port, "6000");
    EXPECT_EQ(options.grpc.grpc_num_cqs, 2);
    EXPECT_EQ(options.grpc.grpc_min_pollers, 3);
    EXPECT_EQ(options.grpc.grpc_max_pollers, 3);
    EXPECT_EQ(options.grpc.max_receive_message_mb, 128);
    EXPECT_EQ(options.grpc.max_send_message_mb, 32);
    EXPECT_EQ(options.grpc.stats_log_interval_seconds, 0);
    EXPECT_EQ(options.grpc.slow_request_ms, 0);
    EXPECT_EQ(options.auth.token, "token-123");
    EXPECT_EQ(options.auth_source, "command-line");
    EXPECT_EQ(options.auth.metadata_key, "x-test-auth");
    EXPECT_EQ(options.limits.max_image_bytes, 8u * 1024u * 1024u);
    EXPECT_EQ(options.limits.max_context_size, 2048);
    EXPECT_TRUE(options.vlm_cache.enabled);
    EXPECT_TRUE(options.vlm_cache.persist);
    EXPECT_EQ(options.vlm_cache.cache_dir, std::filesystem::path("cache/test"));
    EXPECT_EQ(options.vram.warning_free_bytes, 256u * 1024u * 1024u);
    EXPECT_EQ(options.vram.unload_free_bytes, 128u * 1024u * 1024u);
}

TEST(ConfigOptionParserTest, LoadsConfigFileBeforeCliOverrides) {
    ScopedTempDirectory temp("agent_config_test");
    const auto config_path = WriteFile(
        temp.path() / "server.json",
        R"({
            "models": {
                "llm": "config.gguf",
                "bert": "config-bert.onnx",
                "n_gpu_layers": 4,
                "provider": "cpu"
            },
            "grpc": {
                "host": "127.0.0.2",
                "port": "50099",
                "max_receive_message_mb": 64,
                "num_cqs": 1
            },
            "auth": {
                "metadata_key": "x-config-auth"
            },
            "limits": {
                "min_context_size": 128,
                "max_context_size": 1024,
                "max_image_mb": 2
            },
            "vlm_cache": {
                "enabled": true,
                "persist": false,
                "dir": "cache/from-config",
                "vector": {
                    "enabled": true,
                    "sim_threshold_high": 0.95,
                    "dir": "cache/from-config/vectors"
                }
            },
            "vram_guard": {
                "warning_free_mb": 512,
                "unload_free_mb": 256
            }
        })");

    ArgvBuilder argv(std::vector<std::string>{
        "server",
        "--config",
        config_path.string(),
        "--llm", "cli.gguf",
        "--port", "50100",
        "--max-context-size", "2048"
    });

    auto options = ParseMultimodalOptions(argv.argc(), argv.argv());

    EXPECT_EQ(options.llm_model, "cli.gguf");
    EXPECT_EQ(options.bert_model, "config-bert.onnx");
    EXPECT_EQ(options.n_gpu_layers, 4);
    EXPECT_EQ(options.bert_runtime.execution_provider, "cpu");
    EXPECT_EQ(options.grpc.host, "127.0.0.2");
    EXPECT_EQ(options.grpc.port, "50100");
    EXPECT_EQ(options.grpc.max_receive_message_mb, 64);
    EXPECT_EQ(options.grpc.grpc_num_cqs, 1);
    EXPECT_EQ(options.auth.metadata_key, "x-config-auth");
    EXPECT_EQ(options.limits.max_context_size, 2048);
    EXPECT_EQ(options.limits.max_image_bytes, 2u * 1024u * 1024u);
    EXPECT_TRUE(options.vlm_cache.enabled);
    EXPECT_FALSE(options.vlm_cache.persist);
    EXPECT_EQ(options.vlm_cache.cache_dir, std::filesystem::path("cache/from-config"));
    EXPECT_TRUE(options.vlm_cache_vector.enabled);
    EXPECT_FLOAT_EQ(options.vlm_cache_vector.sim_threshold_high, 0.95f);
    EXPECT_EQ(options.vlm_cache_vector.vector_dir, std::filesystem::path("cache/from-config/vectors"));
    EXPECT_EQ(options.vram.warning_free_bytes, 512u * 1024u * 1024u);
    EXPECT_EQ(options.vram.unload_free_bytes, 256u * 1024u * 1024u);
}

TEST(ConfigOptionParserTest, ResolvesAuthTokenFromFile) {
    ScopedTempDirectory temp("agent_auth_test");
    const auto token_path = WriteFile(temp.path() / "token.txt", "\xEF\xBB\xBF  file-token  \n");

    ArgvBuilder argv(std::vector<std::string>{
        "server",
        "--llm", "llm.gguf",
        "--auth-token-file",
        token_path.string()
    });

    auto options = ParseMultimodalOptions(argv.argc(), argv.argv());

    EXPECT_EQ(options.auth.token, "file-token");
    EXPECT_EQ(options.auth_source, "file");
}

TEST(ConfigOptionParserTest, RejectsMissingRequiredLlm) {
    EXPECT_THROW(Parse({"server", "--bert", "bert.onnx"}), std::runtime_error);
}

TEST(ConfigOptionParserTest, RejectsInvalidSectionValidation) {
    EXPECT_THROW(
        Parse({
            "server",
            "--llm", "llm.gguf",
            "--max-context-size", "64"
        }),
        std::runtime_error);

    EXPECT_THROW(
        Parse({
            "server",
            "--llm", "llm.gguf",
            "--vram-warning-free-mb", "128",
            "--vram-unload-free-mb", "256"
        }),
        std::runtime_error);

    EXPECT_THROW(
        Parse({
            "server",
            "--llm", "llm.gguf",
            "--vlm-cache-persist",
            "--vlm-cache-dir", ""
        }),
        std::runtime_error);
}

// ── LLM section ──────────────────────────────────────────────────────────────

namespace {

void UnsetLlmEnv() {
#ifdef _WIN32
    _putenv_s("AGENT_LLM_API_KEY", "");
    _putenv_s("TEST_LLM_KEY", "");
#else
    unsetenv("AGENT_LLM_API_KEY");
    unsetenv("TEST_LLM_KEY");
#endif
}

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

}  // namespace

TEST(ConfigLlmSectionTest, DisabledWhenBaseUrlEmpty) {
    UnsetLlmEnv();
    auto opts = Parse({"server", "--llm", "llm.gguf"});
    EXPECT_TRUE(opts.llm.base_url.empty());
    EXPECT_TRUE(opts.llm.api_key.empty());
}

TEST(ConfigLlmSectionTest, ApiKeyResolvedFromEnv) {
    UnsetLlmEnv();
    SetEnv("AGENT_LLM_API_KEY", "sk-from-env");

    auto opts = Parse({
        "server",
        "--llm", "llm.gguf",
        "--llm-base-url", "https://api.example.com/v1",
    });
    EXPECT_EQ(opts.llm.base_url, "https://api.example.com/v1");
    EXPECT_EQ(opts.llm.api_key, "sk-from-env");

    UnsetLlmEnv();
}

TEST(ConfigLlmSectionTest, CustomEnvVarOverridesDefault) {
    UnsetLlmEnv();
    SetEnv("TEST_LLM_KEY", "sk-custom-env");

    auto opts = Parse({
        "server",
        "--llm", "llm.gguf",
        "--llm-base-url", "https://api.example.com/v1",
        "--llm-api-key-env", "TEST_LLM_KEY",
    });
    EXPECT_EQ(opts.llm.api_key, "sk-custom-env");

    UnsetLlmEnv();
}

TEST(ConfigLlmSectionTest, ApiKeyResolvedFromFileWhenEnvMissing) {
    UnsetLlmEnv();
    ScopedTempDirectory tmp("llm_key");
    auto key_file = tmp.path() / "key.txt";
    WriteFile(key_file, "sk-from-file\n");

    auto opts = Parse({
        "server",
        "--llm", "llm.gguf",
        "--llm-base-url", "https://api.example.com/v1",
        "--llm-api-key-file", key_file.string(),
    });
    EXPECT_EQ(opts.llm.api_key, "sk-from-file");
}

TEST(ConfigLlmSectionTest, EnvTakesPrecedenceOverFile) {
    UnsetLlmEnv();
    SetEnv("AGENT_LLM_API_KEY", "sk-env-wins");

    ScopedTempDirectory tmp("llm_key");
    auto key_file = tmp.path() / "key.txt";
    WriteFile(key_file, "sk-from-file");

    auto opts = Parse({
        "server",
        "--llm", "llm.gguf",
        "--llm-base-url", "https://api.example.com/v1",
        "--llm-api-key-file", key_file.string(),
    });
    EXPECT_EQ(opts.llm.api_key, "sk-env-wins");

    UnsetLlmEnv();
}

TEST(ConfigLlmSectionTest, ThrowsWhenBaseUrlSetButNoKey) {
    UnsetLlmEnv();
    EXPECT_THROW(
        Parse({
            "server",
            "--llm", "llm.gguf",
            "--llm-base-url", "https://api.example.com/v1",
        }),
        std::runtime_error);
}

TEST(ConfigLlmSectionTest, ThrowsWhenKeyFileMissing) {
    UnsetLlmEnv();
    EXPECT_THROW(
        Parse({
            "server",
            "--llm", "llm.gguf",
            "--llm-base-url", "https://api.example.com/v1",
            "--llm-api-key-file", "/nonexistent/path/key.txt",
        }),
        std::runtime_error);
}

TEST(ConfigLlmSectionTest, JsonConfigLoadsLlmSection) {
    UnsetLlmEnv();
    SetEnv("AGENT_LLM_API_KEY", "sk-json-test");

    ScopedTempDirectory tmp("llm_cfg");
    auto config_file = tmp.path() / "config.json";
    WriteFile(config_file, R"({
        "llm": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "timeout_ms": 45000,
            "max_retries": 5,
            "prompts": {
                "memory_extraction": "prompts/memory.txt",
                "fact_distillation": "prompts/fact.txt"
            }
        }
    })");

    auto opts = Parse({
        "server",
        "--llm", "llm.gguf",
        "--config", config_file.string(),
    });

    EXPECT_EQ(opts.llm.base_url, "https://api.deepseek.com/v1");
    EXPECT_EQ(opts.llm.model, "deepseek-chat");
    EXPECT_EQ(opts.llm.timeout_ms, 45000);
    EXPECT_EQ(opts.llm.max_retries, 5);
    EXPECT_EQ(opts.llm.api_key, "sk-json-test");
    EXPECT_EQ(opts.llm.prompts.size(), 2u);
    EXPECT_EQ(opts.llm.prompts["memory_extraction"].string(), "prompts/memory.txt");

    EXPECT_TRUE(opts.config_file_path.is_absolute());

    UnsetLlmEnv();
}

TEST(ConfigLlmSectionTest, CliOverridesJson) {
    UnsetLlmEnv();
    SetEnv("AGENT_LLM_API_KEY", "sk-test");

    ScopedTempDirectory tmp("llm_cfg");
    auto config_file = tmp.path() / "config.json";
    WriteFile(config_file, R"({
        "llm": {
            "base_url": "https://json.example.com/v1",
            "model": "json-model"
        }
    })");

    auto opts = Parse({
        "server",
        "--llm", "llm.gguf",
        "--config", config_file.string(),
        "--llm-model", "cli-model",
        "--llm-timeout", "12345",
    });

    EXPECT_EQ(opts.llm.base_url, "https://json.example.com/v1");
    EXPECT_EQ(opts.llm.model, "cli-model");
    EXPECT_EQ(opts.llm.timeout_ms, 12345);

    UnsetLlmEnv();
}

