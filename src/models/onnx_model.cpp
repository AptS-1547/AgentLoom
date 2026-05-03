/**
 * @file onnx_model.cpp
 * @brief ONNX Runtime BERT 模型实现
 */

#include "onnx_model.h"

#include <algorithm>
#include <cctype>
#include <numeric>
#include <sstream>
#include <thread>

#include <spdlog/spdlog.h>

#include "logger.h"

namespace bert {

namespace {

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

std::string JoinProviders(const std::vector<std::string>& providers) {
    if (providers.empty()) {
        return "none";
    }

    std::ostringstream oss;
    for (size_t i = 0; i < providers.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << providers[i];
    }
    return oss.str();
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

} // namespace

// 内部实现结构
struct OnnxBERTModel::Impl {
    // ONNX Runtime 环境（进程级单例）
    static Ort::Env& GetEnv() {
        static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "BERTInference");
        return env;
    }

    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memory_info;

    // 输入输出节点名称
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::string> input_name_strings;
    std::vector<std::string> output_name_strings;

    std::string requested_provider = "auto";
    std::string active_provider = "cpu";
    std::string provider_note;
    std::string available_providers;
    int intra_op_num_threads = 0;
    int inter_op_num_threads = 0;

    // 节点维度信息
    const size_t num_emotions = 10;
    const size_t num_behaviors = 12;
    const size_t num_tones = 8;
    const size_t personality_dim = 11;
    const size_t num_length_classes = 3;
};

OnnxBERTModel::OnnxBERTModel() : impl_(std::make_unique<Impl>()) {}

OnnxBERTModel::~OnnxBERTModel() = default;

OnnxBERTModel::OnnxBERTModel(OnnxBERTModel&&) noexcept = default;

OnnxBERTModel& OnnxBERTModel::operator=(OnnxBERTModel&&) noexcept = default;

bool OnnxBERTModel::LoadModel(const std::filesystem::path& model_path,
                              const ModelRuntimeOptions& options) {
    try {
        impl_->session.reset();
        impl_->memory_info.reset();
        impl_->input_names.clear();
        impl_->output_names.clear();
        impl_->input_name_strings.clear();
        impl_->output_name_strings.clear();

        impl_->requested_provider = NormalizeProviderPreference(options.execution_provider);
        impl_->available_providers = JoinProviders(Ort::GetAvailableProviders());
        impl_->active_provider = "cpu";
        impl_->provider_note.clear();
        impl_->intra_op_num_threads = 0;
        impl_->inter_op_num_threads = 0;

        const auto create_session = [&](bool use_cuda) {
            Ort::SessionOptions session_options;
            ApplyCommonSessionOptions(
                session_options,
                options,
                use_cuda,
                impl_->intra_op_num_threads,
                impl_->inter_op_num_threads
            );

            if (use_cuda) {
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = options.cuda_device_id;
                cuda_options.do_copy_in_default_stream = 1;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
            }

            const auto native_model_path = model_path.native();
            impl_->session = std::make_unique<Ort::Session>(
                Impl::GetEnv(),
                native_model_path.c_str(),
                session_options
            );
            impl_->active_provider = use_cuda ? "cuda" : "cpu";
        };

        if (impl_->requested_provider == "cpu") {
            create_session(false);
        } else {
            try {
                create_session(true);
            } catch (const Ort::Exception& e) {
                if (!options.allow_cpu_fallback) {
                    throw;
                }

                impl_->provider_note =
                    std::string("CUDA provider unavailable, falling back to CPU: ") + e.what();
                create_session(false);
            }
        }

        // 创建内存信息（CPU）
        impl_->memory_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
        );

        // 获取输入节点名称
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = impl_->session->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = impl_->session->GetInputNameAllocated(i, allocator);
            impl_->input_name_strings.push_back(name.get());
        }
        for (const auto& name : impl_->input_name_strings) {
            impl_->input_names.push_back(name.c_str());
        }

        // 获取输出节点名称
        size_t num_outputs = impl_->session->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = impl_->session->GetOutputNameAllocated(i, allocator);
            impl_->output_name_strings.push_back(name.get());
        }
        for (const auto& name : impl_->output_name_strings) {
            impl_->output_names.push_back(name.c_str());
        }

        loaded_ = true;
        spdlog::info(
            "[BERT] ONNX model loaded: {} inputs, {} outputs, provider={}, intra_op={}, inter_op={}",
            impl_->input_names.size(),
            impl_->output_names.size(),
            impl_->active_provider,
            impl_->intra_op_num_threads,
            impl_->inter_op_num_threads
        );
        if (!impl_->provider_note.empty()) {
            spdlog::warn("[BERT] {}", impl_->provider_note);
        }
        return true;

    } catch (const Ort::Exception& e) {
        spdlog::error("[BERT] ONNX Runtime error: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        spdlog::error("[BERT] Error loading model: {}", e.what());
        return false;
    }
}

InferenceResult OnnxBERTModel::Predict(
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& attention_mask,
    const std::vector<float>& personality) {

    if (!loaded_) {
        return InferenceResult{ .success = false, .error_message = "Model not loaded" };
    }

    // 验证输入维度
    if (input_ids.size() != attention_mask.size()) {
        return InferenceResult{ .success = false, .error_message = "input_ids and attention_mask size mismatch" };
    }
    if (personality.size() != impl_->personality_dim) {
        return InferenceResult{ .success = false, .error_message = "personality dim must be 11" };
    }

    size_t seq_len = input_ids.size();

    try {
        // 构建输入 tensors
        std::array<int64_t, 2> input_shape = {1, static_cast<int64_t>(seq_len)};
        std::array<int64_t, 2> personality_shape = {1, static_cast<int64_t>(impl_->personality_dim)};

        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            *impl_->memory_info,
            const_cast<int64_t*>(input_ids.data()),
            input_ids.size(),
            input_shape.data(),
            input_shape.size()
        );

        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            *impl_->memory_info,
            const_cast<int64_t*>(attention_mask.data()),
            attention_mask.size(),
            input_shape.data(),
            input_shape.size()
        );

        Ort::Value personality_tensor = Ort::Value::CreateTensor<float>(
            *impl_->memory_info,
            const_cast<float*>(personality.data()),
            personality.size(),
            personality_shape.data(),
            personality_shape.size()
        );

        // 输入数组（按 proto 定义的顺序）
        std::array<Ort::Value, 3> input_tensors = {
            std::move(input_ids_tensor),
            std::move(attention_mask_tensor),
            std::move(personality_tensor)
        };

        // 运行推理
        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            impl_->output_names.data(),
            impl_->output_names.size()
        );

        // 解析输出
        // output_tensors 顺序: emotion_logits, behavior_logits, tone_logits, intensity, response_length
        InferenceResult result;
        result.success = true;

        // emotion_logits [1, 10]
        float* emotion_data = output_tensors[0].GetTensorMutableData<float>();
        std::copy_n(emotion_data, result.emotion_logits.size(), result.emotion_logits.begin());

        // behavior_logits [1, 12]
        float* behavior_data = output_tensors[1].GetTensorMutableData<float>();
        std::copy_n(behavior_data, result.behavior_logits.size(), result.behavior_logits.begin());

        // tone_logits [1, 8]
        float* tone_data = output_tensors[2].GetTensorMutableData<float>();
        std::copy_n(tone_data, result.tone_logits.size(), result.tone_logits.begin());

        // intensity [1, 1]
        float* intensity_data = output_tensors[3].GetTensorMutableData<float>();
        result.intensity = intensity_data[0];

        // response_length [1, 3]
        float* length_data = output_tensors[4].GetTensorMutableData<float>();
        std::copy_n(length_data, result.response_length_logits.size(), result.response_length_logits.begin());

        return result;

    } catch (const Ort::Exception& e) {
        return InferenceResult{ .success = false, .error_message = e.what() };
    } catch (const std::exception& e) {
        return InferenceResult{ .success = false, .error_message = e.what() };
    }
}

std::vector<InferenceResult> OnnxBERTModel::PredictBatch(
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& attention_mask,
    const std::vector<float>& personality,
    size_t batch_size,
    size_t seq_len) {

    if (!loaded_) {
        return {};
    }

    // 验证输入尺寸
    if (input_ids.size() != batch_size * seq_len ||
        attention_mask.size() != batch_size * seq_len ||
        personality.size() != batch_size * impl_->personality_dim) {
        spdlog::error("[BERT] Batch input size mismatch");
        return {};
    }

    try {
        // 构建 batch 输入 tensors
        std::array<int64_t, 2> input_shape = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(seq_len)
        };
        std::array<int64_t, 2> personality_shape = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(impl_->personality_dim)
        };

        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            *impl_->memory_info,
            const_cast<int64_t*>(input_ids.data()),
            input_ids.size(),
            input_shape.data(),
            input_shape.size()
        );

        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            *impl_->memory_info,
            const_cast<int64_t*>(attention_mask.data()),
            attention_mask.size(),
            input_shape.data(),
            input_shape.size()
        );

        Ort::Value personality_tensor = Ort::Value::CreateTensor<float>(
            *impl_->memory_info,
            const_cast<float*>(personality.data()),
            personality.size(),
            personality_shape.data(),
            personality_shape.size()
        );

        std::array<Ort::Value, 3> input_tensors = {
            std::move(input_ids_tensor),
            std::move(attention_mask_tensor),
            std::move(personality_tensor)
        };

        // 运行推理
        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            impl_->output_names.data(),
            impl_->output_names.size()
        );

        // 解析 batch 输出
        std::vector<InferenceResult> results;
        results.reserve(batch_size);

        // 获取各输出的数据指针
        float* emotion_data = output_tensors[0].GetTensorMutableData<float>();
        float* behavior_data = output_tensors[1].GetTensorMutableData<float>();
        float* tone_data = output_tensors[2].GetTensorMutableData<float>();
        float* intensity_data = output_tensors[3].GetTensorMutableData<float>();
        float* length_data = output_tensors[4].GetTensorMutableData<float>();

        for (size_t i = 0; i < batch_size; ++i) {
            InferenceResult result;
            result.success = true;

            // emotion_logits [batch, 10]
            std::copy_n(
                emotion_data + i * result.emotion_logits.size(),
                result.emotion_logits.size(),
                result.emotion_logits.begin()
            );

            // behavior_logits [batch, 12]
            std::copy_n(
                behavior_data + i * result.behavior_logits.size(),
                result.behavior_logits.size(),
                result.behavior_logits.begin()
            );

            // tone_logits [batch, 8]
            std::copy_n(
                tone_data + i * result.tone_logits.size(),
                result.tone_logits.size(),
                result.tone_logits.begin()
            );

            // intensity [batch, 1]
            result.intensity = intensity_data[i];

            // response_length [batch, 3]
            std::copy_n(
                length_data + i * result.response_length_logits.size(),
                result.response_length_logits.size(),
                result.response_length_logits.begin()
            );

            results.push_back(std::move(result));
        }

        return results;

    } catch (const Ort::Exception& e) {
        spdlog::error("[BERT] Batch inference error: {}", e.what());
        return {};
    } catch (const std::exception& e) {
        spdlog::error("[BERT] Batch inference error: {}", e.what());
        return {};
    }
}

std::string OnnxBERTModel::GetInfo() const {
    if (!loaded_) {
        return "Model not loaded";
    }
    std::ostringstream oss;
    oss << "ONNX BERT Model loaded with "
        << impl_->input_names.size() << " inputs, "
        << impl_->output_names.size() << " outputs"
        << ", requested_provider=" << impl_->requested_provider
        << ", active_provider=" << impl_->active_provider
        << ", available_providers=[" << impl_->available_providers << "]"
        << ", intra_op=" << impl_->intra_op_num_threads
        << ", inter_op=" << impl_->inter_op_num_threads;
    if (!impl_->provider_note.empty()) {
        oss << ", note=" << impl_->provider_note;
    }
    return oss.str();
}

std::string OnnxBERTModel::GetActiveExecutionProvider() const {
    if (!loaded_) {
        return "unloaded";
    }
    return impl_->active_provider;
}

} // namespace bert
