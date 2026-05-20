/**
 * @file onnx_model.cpp
 * @brief ONNX Runtime BERT 模型实现
 */

#include "onnx_model.h"

#include "onnx_session_utils.h"

#include <algorithm>
#include <numeric>
#include <sstream>

#include <spdlog/spdlog.h>

namespace bert {

// 内部实现结构
struct OnnxBERTModel::Impl {
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memory_info;

    // 输入输出节点名称
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::string> input_name_strings;
    std::vector<std::string> output_name_strings;

    OnnxSessionInfo session_info;

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

        auto bundle = CreateOnnxSessionBundle(model_path, options, "BERT");
        impl_->session = std::move(bundle.session);
        impl_->memory_info = std::move(bundle.memory_info);
        impl_->input_name_strings = std::move(bundle.input_name_strings);
        impl_->output_name_strings = std::move(bundle.output_name_strings);
        impl_->input_names = std::move(bundle.input_names);
        impl_->output_names = std::move(bundle.output_names);
        impl_->session_info = std::move(bundle.info);

        // After moving input_name_strings/output_name_strings, the c_str()
        // pointers we previously pushed into bundle.input_names/output_names
        // still point to the strings now held by impl_, since std::string
        // moves preserve the underlying buffer for SSO-fitting and heap
        // strings alike when source is destroyed. To be defensive, rebuild
        // them from the moved-in name vectors.
        impl_->input_names.clear();
        impl_->input_names.reserve(impl_->input_name_strings.size());
        for (const auto& s : impl_->input_name_strings) {
            impl_->input_names.push_back(s.c_str());
        }
        impl_->output_names.clear();
        impl_->output_names.reserve(impl_->output_name_strings.size());
        for (const auto& s : impl_->output_name_strings) {
            impl_->output_names.push_back(s.c_str());
        }

        loaded_ = true;
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
        << ", requested_provider=" << impl_->session_info.requested_provider
        << ", active_provider=" << impl_->session_info.active_provider
        << ", available_providers=[" << impl_->session_info.available_providers << "]"
        << ", intra_op=" << impl_->session_info.intra_op_num_threads
        << ", inter_op=" << impl_->session_info.inter_op_num_threads;
    if (!impl_->session_info.provider_note.empty()) {
        oss << ", note=" << impl_->session_info.provider_note;
    }
    return oss.str();
}

std::string OnnxBERTModel::GetActiveExecutionProvider() const {
    if (!loaded_) {
        return "unloaded";
    }
    return impl_->session_info.active_provider;
}

} // namespace bert
