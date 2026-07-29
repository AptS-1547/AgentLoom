#include "onnx_text_embedding_model.h"

#include "onnx_session_utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>

namespace vector {

namespace {

std::string LowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

enum class InputKind {
    Unknown,
    InputIds,
    AttentionMask,
    TokenTypeIds,
};

InputKind ClassifyInputName(const std::string& name) {
    const std::string lower = LowerCopy(name);
    if (lower == "input_ids" || lower == "input.1" || lower == "input_ids:0") {
        return InputKind::InputIds;
    }
    if (lower == "attention_mask" || lower == "attention_mask:0") {
        return InputKind::AttentionMask;
    }
    if (lower == "token_type_ids" || lower == "token_type_ids:0" ||
        lower == "segment_ids") {
        return InputKind::TokenTypeIds;
    }
    return InputKind::Unknown;
}

bool IsLikelyPooledOutputName(const std::string& name) {
    const std::string lower = LowerCopy(name);
    return lower.find("sentence_embedding") != std::string::npos ||
           lower.find("pooler_output") != std::string::npos ||
           lower.find("pooled") != std::string::npos ||
           lower == "embeddings";
}

bool IsLikelyHiddenStateName(const std::string& name) {
    const std::string lower = LowerCopy(name);
    return lower.find("last_hidden_state") != std::string::npos ||
           lower.find("hidden_state") != std::string::npos ||
           lower == "output" || lower == "logits";
}

} // namespace

struct OnnxTextEmbeddingModel::Impl {
    EmbeddingModelOptions options;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> memory_info;
    std::vector<std::string> input_name_strings;
    std::vector<std::string> output_name_strings;
    std::vector<const char*> input_name_cstrs;
    std::vector<const char*> output_name_cstrs;
    bert::OnnxSessionInfo session_info;

    // Mapping: for each session input slot, which input kind to fill in.
    std::vector<InputKind> input_kinds;

    // Output selection: which session output index to read for embeddings.
    std::size_t embedding_output_index = 0;
    bool output_is_already_pooled = false;

    // Detected hidden dim. 0 means "not yet known" -> learn on first run.
    std::atomic<std::size_t> hidden_size{0};
};

OnnxTextEmbeddingModel::OnnxTextEmbeddingModel() : impl_(std::make_unique<Impl>()) {}
OnnxTextEmbeddingModel::~OnnxTextEmbeddingModel() = default;
OnnxTextEmbeddingModel::OnnxTextEmbeddingModel(OnnxTextEmbeddingModel&&) noexcept = default;
OnnxTextEmbeddingModel& OnnxTextEmbeddingModel::operator=(OnnxTextEmbeddingModel&&) noexcept = default;

std::size_t OnnxTextEmbeddingModel::Dimension() const noexcept {
    return impl_ ? impl_->hidden_size.load(std::memory_order_relaxed) : 0;
}

PoolingStrategy OnnxTextEmbeddingModel::Pooling() const noexcept {
    return impl_ ? impl_->options.pooling : PoolingStrategy::Mean;
}

bool OnnxTextEmbeddingModel::Normalized() const noexcept {
    return impl_ ? impl_->options.normalize : false;
}

core::Result<std::unique_ptr<OnnxTextEmbeddingModel>> OnnxTextEmbeddingModel::Load(
    EmbeddingModelOptions options) {

    if (options.model_path.empty()) {
        return core::Status(core::ErrorCode::InvalidArgument,
                            "OnnxTextEmbeddingModel::Load: model_path is empty");
    }

    auto model = std::make_unique<OnnxTextEmbeddingModel>();
    auto& impl = *model->impl_;

    bert::ModelRuntimeOptions rt;
    rt.execution_provider = options.execution_provider;
    rt.allow_cpu_fallback = options.allow_cpu_fallback;
    rt.cuda_device_id = options.cuda_device_id;
    rt.intra_op_num_threads = options.intra_op_num_threads;
    rt.inter_op_num_threads = options.inter_op_num_threads;

    try {
        auto bundle = bert::CreateOnnxSessionBundle(options.model_path, rt, "TextEmbedding");
        impl.session = std::move(bundle.session);
        impl.memory_info = std::move(bundle.memory_info);
        impl.input_name_strings = std::move(bundle.input_name_strings);
        impl.output_name_strings = std::move(bundle.output_name_strings);
        impl.session_info = std::move(bundle.info);
    } catch (const Ort::Exception& e) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            std::string("ONNX session creation failed: ") + e.what());
    } catch (const std::exception& e) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            std::string("ONNX session creation failed: ") + e.what());
    }

    impl.input_name_cstrs.reserve(impl.input_name_strings.size());
    for (const auto& s : impl.input_name_strings) {
        impl.input_name_cstrs.push_back(s.c_str());
    }
    impl.output_name_cstrs.reserve(impl.output_name_strings.size());
    for (const auto& s : impl.output_name_strings) {
        impl.output_name_cstrs.push_back(s.c_str());
    }

    impl.input_kinds.reserve(impl.input_name_strings.size());
    bool has_input_ids = false;
    bool has_attention = false;
    bool has_token_types = false;
    for (const auto& name : impl.input_name_strings) {
        auto kind = ClassifyInputName(name);
        if (kind == InputKind::InputIds) has_input_ids = true;
        if (kind == InputKind::AttentionMask) has_attention = true;
        if (kind == InputKind::TokenTypeIds) has_token_types = true;
        impl.input_kinds.push_back(kind);
    }
    if (!has_input_ids) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "OnnxTextEmbeddingModel::Load: model has no input_ids input");
    }
    if (!has_attention && options.pooling == PoolingStrategy::Mean) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "OnnxTextEmbeddingModel::Load: Mean pooling needs attention_mask input");
    }
    if (options.require_token_type_ids && !has_token_types) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "OnnxTextEmbeddingModel::Load: token_type_ids required but not in model");
    }

    if (impl.output_name_strings.empty()) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "OnnxTextEmbeddingModel::Load: model has no outputs");
    }
    impl.embedding_output_index = 0;
    for (std::size_t i = 0; i < impl.output_name_strings.size(); ++i) {
        if (IsLikelyPooledOutputName(impl.output_name_strings[i])) {
            impl.embedding_output_index = i;
            impl.output_is_already_pooled = true;
            break;
        }
    }
    if (!impl.output_is_already_pooled) {
        for (std::size_t i = 0; i < impl.output_name_strings.size(); ++i) {
            if (IsLikelyHiddenStateName(impl.output_name_strings[i])) {
                impl.embedding_output_index = i;
                break;
            }
        }
    }

    impl.options = std::move(options);
    return model;
}

core::Result<EmbeddingBatch> OnnxTextEmbeddingModel::Embed(const TokenizedBatch& batch) const {
    if (!impl_ || !impl_->session) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "OnnxTextEmbeddingModel::Embed: session not loaded");
    }
    if (batch.batch_size == 0 || batch.sequence_length == 0) {
        return core::Status(core::ErrorCode::InvalidArgument,
                            "OnnxTextEmbeddingModel::Embed: empty batch");
    }
    const std::size_t expected = batch.batch_size * batch.sequence_length;
    if (batch.input_ids.size() != expected) {
        return core::Status(core::ErrorCode::InvalidArgument,
                            "OnnxTextEmbeddingModel::Embed: input_ids size mismatch");
    }
    if (batch.attention_mask.size() != expected) {
        return core::Status(core::ErrorCode::InvalidArgument,
                            "OnnxTextEmbeddingModel::Embed: attention_mask size mismatch");
    }

    std::array<std::int64_t, 2> shape{
        static_cast<std::int64_t>(batch.batch_size),
        static_cast<std::int64_t>(batch.sequence_length)
    };

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(impl_->input_kinds.size());

    std::vector<std::int64_t> token_type_buf;
    for (auto kind : impl_->input_kinds) {
        const std::int64_t* src = nullptr;
        switch (kind) {
            case InputKind::InputIds:
                src = batch.input_ids.data();
                break;
            case InputKind::AttentionMask:
                src = batch.attention_mask.data();
                break;
            case InputKind::TokenTypeIds:
                if (batch.token_type_ids.size() == expected) {
                    src = batch.token_type_ids.data();
                } else {
                    token_type_buf.assign(expected, 0);
                    src = token_type_buf.data();
                }
                break;
            case InputKind::Unknown:
                return core::Status(core::ErrorCode::FailedPrecondition,
                                    "OnnxTextEmbeddingModel::Embed: unsupported model input slot");
        }
        Ort::Value tensor = Ort::Value::CreateTensor<std::int64_t>(
            *impl_->memory_info,
            const_cast<std::int64_t*>(src),
            expected,
            shape.data(),
            shape.size());
        input_tensors.push_back(std::move(tensor));
    }

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_name_cstrs.data(),
            input_tensors.data(),
            input_tensors.size(),
            impl_->output_name_cstrs.data(),
            impl_->output_name_cstrs.size()
        );
    } catch (const Ort::Exception& e) {
        return core::Status(core::ErrorCode::InternalError,
                            std::string("ONNX Run failed: ") + e.what());
    }

    if (impl_->embedding_output_index >= output_tensors.size()) {
        return core::Status(core::ErrorCode::InternalError,
                            "embedding output index out of range");
    }
    Ort::Value& output = output_tensors[impl_->embedding_output_index];
    auto type_shape = output.GetTensorTypeAndShapeInfo();
    auto out_shape = type_shape.GetShape();

    EmbeddingBatch result;

    if (impl_->output_is_already_pooled || out_shape.size() == 2) {
        if (out_shape.size() != 2) {
            return core::Status(core::ErrorCode::InternalError,
                                "pooled output is not rank-2");
        }
        const std::size_t bs = static_cast<std::size_t>(out_shape[0]);
        const std::size_t dim = static_cast<std::size_t>(out_shape[1]);
        if (bs != batch.batch_size) {
            return core::Status(core::ErrorCode::InternalError,
                                "pooled output batch size mismatch");
        }
        result.batch_size = bs;
        result.dimension = dim;
        result.embeddings.assign(
            output.GetTensorMutableData<float>(),
            output.GetTensorMutableData<float>() + bs * dim);
    } else if (out_shape.size() == 3) {
        const std::size_t bs = static_cast<std::size_t>(out_shape[0]);
        const std::size_t seq = static_cast<std::size_t>(out_shape[1]);
        const std::size_t hidden = static_cast<std::size_t>(out_shape[2]);
        if (bs != batch.batch_size || seq != batch.sequence_length) {
            return core::Status(core::ErrorCode::InternalError,
                                "hidden state shape mismatches input");
        }
        result.batch_size = bs;
        result.dimension = hidden;
        result.embeddings.resize(bs * hidden);

        const float* hidden_data = output.GetTensorMutableData<float>();
        core::Status status;
        switch (impl_->options.pooling) {
            case PoolingStrategy::Mean:
                status = pooling::MeanPool(
                    hidden_data, batch.attention_mask.data(),
                    bs, seq, hidden, result.embeddings.data());
                break;
            case PoolingStrategy::Cls:
                status = pooling::ClsPool(
                    hidden_data, bs, seq, hidden, result.embeddings.data());
                break;
        }
        if (!status.ok()) {
            return status;
        }
    } else {
        return core::Status(core::ErrorCode::InternalError,
                            "unsupported output tensor rank");
    }

    if (impl_->options.normalize) {
        pooling::L2NormalizeRows(result.embeddings.data(), result.batch_size, result.dimension);
    }

    if (impl_->options.expected_dimension != 0 &&
        impl_->options.expected_dimension != result.dimension) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "embedding dimension differs from expected");
    }

    impl_->hidden_size.store(result.dimension, std::memory_order_relaxed);
    return result;
}

std::string OnnxTextEmbeddingModel::GetInfo() const {
    if (!impl_ || !impl_->session) {
        return "OnnxTextEmbeddingModel not loaded";
    }
    std::ostringstream oss;
    oss << "OnnxTextEmbeddingModel("
        << "inputs=" << impl_->input_name_strings.size()
        << ", outputs=" << impl_->output_name_strings.size()
        << ", pooled_output=" << (impl_->output_is_already_pooled ? "yes" : "no")
        << ", pooling=" << (impl_->options.pooling == PoolingStrategy::Mean ? "mean" : "cls")
        << ", normalize=" << (impl_->options.normalize ? "true" : "false")
        << ", provider=" << impl_->session_info.active_provider
        << ", hidden=" << impl_->hidden_size.load(std::memory_order_relaxed)
        << ")";
    return oss.str();
}

std::string OnnxTextEmbeddingModel::GetActiveExecutionProvider() const {
    if (!impl_ || !impl_->session) {
        return "unloaded";
    }
    return impl_->session_info.active_provider;
}

} // namespace vector
