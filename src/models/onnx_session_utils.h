#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace bert {

struct ModelRuntimeOptions {
    std::string execution_provider = "auto";
    bool allow_cpu_fallback = true;
    int cuda_device_id = 0;
    int intra_op_num_threads = 0;
    int inter_op_num_threads = 0;
    bool enable_cpu_mem_arena = true;
    bool enable_mem_pattern = true;
};

class OnnxEnv {
public:
    static Ort::Env& Instance();

private:
    OnnxEnv() = delete;
};

struct OnnxSessionInfo {
    std::string requested_provider = "auto";
    std::string active_provider = "cpu";
    std::string provider_note;
    std::string available_providers;
    int intra_op_num_threads = 0;
    int inter_op_num_threads = 0;
};

struct OnnxSessionBundle {
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memory_info;
    std::vector<std::string> input_name_strings;
    std::vector<std::string> output_name_strings;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    OnnxSessionInfo info;
};

OnnxSessionBundle CreateOnnxSessionBundle(
    const std::filesystem::path& model_path,
    const ModelRuntimeOptions& options,
    const std::string& logger_tag);

std::string NormalizeProviderPreference(std::string provider);
std::string JoinProviders(const std::vector<std::string>& providers);

} // namespace bert
