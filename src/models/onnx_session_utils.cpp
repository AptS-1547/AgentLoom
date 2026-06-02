#include "onnx_session_utils.h"

#include "../core/logger_adapter.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <thread>

namespace bert {

namespace {

static core::LoggerAdapter logger = core::LoggerAdapter::ForModule("models");

int ResolveDefaultIntraOpThreads(const ModelRuntimeOptions& options, bool use_cuda) {
    if (options.intra_op_num_threads > 0) {
        return options.intra_op_num_threads;
    }
    if (use_cuda) {
        return 1;
    }
    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    return static_cast<int>(std::clamp(hw_threads / 2, 1u, 8u));
}

int ResolveDefaultInterOpThreads(const ModelRuntimeOptions& options) {
    if (options.inter_op_num_threads > 0) {
        return options.inter_op_num_threads;
    }
    return 1;
}

void ApplyCommonSessionOptions(Ort::SessionOptions& session_options,
                               const ModelRuntimeOptions& options,
                               bool use_cuda,
                               int& resolved_intra_op,
                               int& resolved_inter_op) {
    resolved_intra_op = ResolveDefaultIntraOpThreads(options, use_cuda);
    resolved_inter_op = ResolveDefaultInterOpThreads(options);

    session_options.SetIntraOpNumThreads(resolved_intra_op);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    if (resolved_inter_op > 1) {
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        session_options.SetInterOpNumThreads(resolved_inter_op);
    }
    if (!options.enable_cpu_mem_arena) {
        session_options.DisableCpuMemArena();
    }
    if (!options.enable_mem_pattern) {
        session_options.DisableMemPattern();
    }
}

void PopulateIoNames(OnnxSessionBundle& bundle) {
    Ort::AllocatorWithDefaultOptions allocator;

    const std::size_t num_inputs = bundle.session->GetInputCount();
    for (std::size_t i = 0; i < num_inputs; ++i) {
        auto name = bundle.session->GetInputNameAllocated(i, allocator);
        bundle.input_name_strings.emplace_back(name.get());
    }
    bundle.input_names.reserve(num_inputs);
    for (const auto& s : bundle.input_name_strings) {
        bundle.input_names.push_back(s.c_str());
    }

    const std::size_t num_outputs = bundle.session->GetOutputCount();
    for (std::size_t i = 0; i < num_outputs; ++i) {
        auto name = bundle.session->GetOutputNameAllocated(i, allocator);
        bundle.output_name_strings.emplace_back(name.get());
    }
    bundle.output_names.reserve(num_outputs);
    for (const auto& s : bundle.output_name_strings) {
        bundle.output_names.push_back(s.c_str());
    }
}

} // namespace

Ort::Env& OnnxEnv::Instance() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "AgentBackendOnnx");
    return env;
}

std::string NormalizeProviderPreference(std::string provider) {
    std::transform(
        provider.begin(),
        provider.end(),
        provider.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
    );
    if (provider.empty()) {
        return "auto";
    }
    if (provider == "cpu" || provider == "cuda" || provider == "auto") {
        return provider;
    }
    return "auto";
}

std::string JoinProviders(const std::vector<std::string>& providers) {
    if (providers.empty()) {
        return "none";
    }
    std::ostringstream oss;
    for (size_t i = 0; i < providers.size(); ++i) {
        if (i != 0) oss << ", ";
        oss << providers[i];
    }
    return oss.str();
}

OnnxSessionBundle CreateOnnxSessionBundle(
    const std::filesystem::path& model_path,
    const ModelRuntimeOptions& options,
    const std::string& logger_tag) {

    OnnxSessionBundle bundle;
    bundle.info.requested_provider = NormalizeProviderPreference(options.execution_provider);
    bundle.info.available_providers = JoinProviders(Ort::GetAvailableProviders());

    const auto create_session = [&](bool use_cuda) {
        Ort::SessionOptions session_options;
        ApplyCommonSessionOptions(
            session_options,
            options,
            use_cuda,
            bundle.info.intra_op_num_threads,
            bundle.info.inter_op_num_threads
        );

        if (use_cuda) {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = options.cuda_device_id;
            cuda_options.do_copy_in_default_stream = 1;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }

        const auto native_model_path = model_path.native();
        bundle.session = std::make_unique<Ort::Session>(
            OnnxEnv::Instance(),
            native_model_path.c_str(),
            session_options
        );
        bundle.info.active_provider = use_cuda ? "cuda" : "cpu";
    };

    if (bundle.info.requested_provider == "cpu") {
        create_session(false);
    } else {
        try {
            create_session(true);
        } catch (const Ort::Exception& e) {
            if (!options.allow_cpu_fallback) {
                throw;
            }
            bundle.info.provider_note =
                std::string("CUDA provider unavailable, falling back to CPU: ") + e.what();
            create_session(false);
        }
    }

    bundle.memory_info = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
    );
    PopulateIoNames(bundle);

    logger.info(
        "[{}] ONNX session loaded: {} inputs, {} outputs, provider={}, intra_op={}, inter_op={}",
        logger_tag,
        bundle.input_names.size(),
        bundle.output_names.size(),
        bundle.info.active_provider,
        bundle.info.intra_op_num_threads,
        bundle.info.inter_op_num_threads
    );
    if (!bundle.info.provider_note.empty()) {
        logger.warn("[{}] {}", logger_tag, bundle.info.provider_note);
    }

    return bundle;
}

} // namespace bert
