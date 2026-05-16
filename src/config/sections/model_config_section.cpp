#include "config_section.h"

#include <stdexcept>

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(ModelsConfigSection, "models")
    CONFIG_CLI_STRING(kBert, "--bert");
    CONFIG_CLI_STRING(kVit, "--vit");
    CONFIG_CLI_STRING(kLlm, "--llm");
    CONFIG_CLI_STRING(kMmproj, "--mmproj");
    CONFIG_CLI_STRING(kGpuLayers, "--ngl");
    CONFIG_CLI_STRING(kProvider, "--provider");
    CONFIG_CLI_STRING(kCudaDevice, "--cuda-device");
    void Validate(MultimodalServerOptions& options) const override;
};

void ModelsConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) {
        return;
    }
    SetString(*section, Name(), "llm", options.llm_model);
    SetString(*section, Name(), "mmproj", options.mmproj);
    SetString(*section, Name(), "bert", options.bert_model);
    SetString(*section, Name(), "vit", options.vit_model);
    SetInt(*section, Name(), "n_gpu_layers", options.n_gpu_layers);
    SetString(*section, Name(), "provider", options.bert_runtime.execution_provider);
    SetInt(*section, Name(), "cuda_device", options.bert_runtime.cuda_device_id, 0);
}

bool ModelsConfigSection::LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const {
    CliArgumentParser parser(cursor);
    CONFIG_VALUE_ARG(kBert, value, options.bert_model = *value;)
    CONFIG_VALUE_ARG(kVit, value, options.vit_model = *value;)
    CONFIG_VALUE_ARG(kLlm, value, options.llm_model = *value;)
    CONFIG_VALUE_ARG(kMmproj, value, options.mmproj = *value;)
    CONFIG_VALUE_ARG(kGpuLayers, value, options.n_gpu_layers = ParseCliInt(kGpuLayers, *value);)
    CONFIG_VALUE_ARG(kProvider, value, options.bert_runtime.execution_provider = *value;)
    CONFIG_VALUE_ARG(kCudaDevice, value, options.bert_runtime.cuda_device_id = ParseCliInt(kCudaDevice, *value);)
    return false;
}

void ModelsConfigSection::Validate(MultimodalServerOptions& options) const {
    if (options.llm_model.empty()) {
        throw std::runtime_error("--llm is required");
    }
}

} // namespace

REGISTER_CONFIG_SECTION(ModelsConfigSection)

} // namespace server_config
