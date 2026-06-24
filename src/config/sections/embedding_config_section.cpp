#include "config_section.h"

#include <stdexcept>

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(EmbeddingConfigSection, "embedding")
    CONFIG_CLI_STRING(kTokenizer, "--embedding-tokenizer");
    CONFIG_CLI_STRING(kModel, "--embedding-model");
    CONFIG_CLI_STRING(kProvider, "--embedding-provider");
    CONFIG_CLI_STRING(kCudaDevice, "--embedding-cuda-device");
    CONFIG_CLI_STRING(kIntraOpThreads, "--embedding-intra-threads");
    CONFIG_CLI_STRING(kInterOpThreads, "--embedding-inter-threads");
    CONFIG_CLI_STRING(kPooling, "--embedding-pooling");
    CONFIG_CLI_STRING(kNormalize, "--embedding-normalize");
    CONFIG_CLI_STRING(kDimension, "--embedding-dimension");
    void Validate(MultimodalServerOptions& options) const override;
};

void EmbeddingConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) {
        return;
    }

    SetString(*section, Name(), "tokenizer_path", options.embedding.tokenizer_path);
    SetString(*section, Name(), "onnx_model_path", options.embedding.onnx_model_path);
    if (options.embedding.onnx_model_path.empty()) {
        SetString(*section, Name(), "model_path", options.embedding.onnx_model_path);
    }
    SetString(*section, Name(), "execution_provider", options.embedding.execution_provider);
    SetBool(*section, Name(), "allow_cpu_fallback", options.embedding.allow_cpu_fallback);
    SetInt(*section, Name(), "cuda_device_id", options.embedding.cuda_device_id, 0);
    SetInt(*section, Name(), "intra_op_num_threads", options.embedding.intra_op_num_threads, 0);
    SetInt(*section, Name(), "inter_op_num_threads", options.embedding.inter_op_num_threads, 0);
    SetString(*section, Name(), "pooling_strategy", options.embedding.pooling_strategy);
    SetBool(*section, Name(), "normalize", options.embedding.normalize);
    SetInt(*section, Name(), "expected_dimension", options.embedding.expected_dimension, 0);
    SetBool(*section, Name(), "require_token_type_ids", options.embedding.require_token_type_ids);
}

bool EmbeddingConfigSection::LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const {
    CliArgumentParser parser(cursor);

    CONFIG_VALUE_ARG(kTokenizer, value, options.embedding.tokenizer_path = *value;)
    CONFIG_VALUE_ARG(kModel, value, options.embedding.onnx_model_path = *value;)
    CONFIG_VALUE_ARG(kProvider, value, options.embedding.execution_provider = *value;)
    CONFIG_VALUE_ARG(kCudaDevice, value, {
        options.embedding.cuda_device_id = ParseNonNegativeOption(kCudaDevice, *value);
    })
    CONFIG_VALUE_ARG(kIntraOpThreads, value, {
        options.embedding.intra_op_num_threads = ParseNonNegativeOption(kIntraOpThreads, *value);
    })
    CONFIG_VALUE_ARG(kInterOpThreads, value, {
        options.embedding.inter_op_num_threads = ParseNonNegativeOption(kInterOpThreads, *value);
    })
    CONFIG_VALUE_ARG(kPooling, value, {
        if (*value != "mean" && *value != "cls") {
            throw std::runtime_error("--embedding-pooling must be 'mean' or 'cls'");
        }
        options.embedding.pooling_strategy = *value;
    })
    CONFIG_VALUE_ARG(kNormalize, value, {
        if (*value == "true" || *value == "1") {
            options.embedding.normalize = true;
        } else if (*value == "false" || *value == "0") {
            options.embedding.normalize = false;
        } else {
            throw std::runtime_error("--embedding-normalize must be 'true' or 'false'");
        }
    })
    CONFIG_VALUE_ARG(kDimension, value, {
        options.embedding.expected_dimension = ParsePositiveOption(kDimension, *value);
    })

    return false;
}

void EmbeddingConfigSection::Validate(MultimodalServerOptions& options) const {
    // Embedding 模型是可选的，只有在配置了路径时才验证
    if (!options.embedding.tokenizer_path.empty() || !options.embedding.onnx_model_path.empty()) {
        if (options.embedding.tokenizer_path.empty()) {
            throw std::runtime_error("embedding.tokenizer_path is required when embedding model is configured");
        }
        if (options.embedding.onnx_model_path.empty()) {
            throw std::runtime_error("embedding.onnx_model_path is required when embedding model is configured");
        }

        const auto& pooling = options.embedding.pooling_strategy;
        if (pooling != "mean" && pooling != "cls") {
            throw std::runtime_error("embedding.pooling_strategy must be 'mean' or 'cls'");
        }
    }
}

} // namespace

REGISTER_CONFIG_SECTION(EmbeddingConfigSection)

} // namespace server_config
