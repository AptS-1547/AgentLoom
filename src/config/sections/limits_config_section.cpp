#include "config_section.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(LimitsConfigSection, "limits")
    CONFIG_CLI_STRING(kMaxImageMb, "--max-image-mb");
    CONFIG_CLI_STRING(kMaxImagePixels, "--max-image-pixels");
    CONFIG_CLI_STRING(kMaxImageWidth, "--max-image-width");
    CONFIG_CLI_STRING(kMaxImageHeight, "--max-image-height");
    CONFIG_CLI_STRING(kMaxPromptBytes, "--max-prompt-bytes");
    CONFIG_CLI_STRING(kMaxSequenceLength, "--max-seq-len");
    CONFIG_CLI_STRING(kMaxBatchSize, "--max-batch-size");
    CONFIG_CLI_STRING(kMaxVlmTokens, "--max-vlm-tokens");
    CONFIG_CLI_STRING(kMaxContextSize, "--max-context-size");
    CONFIG_CLI_STRING(kMaxTokenId, "--max-token-id");
    void Validate(MultimodalServerOptions& options) const override;
};

void LimitsConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) {
        return;
    }
    SetMegabytes(*section, Name(), "max_image_mb", options.limits.max_image_bytes, 1);
    SetSize(*section, Name(), "max_image_bytes", options.limits.max_image_bytes, 1);
    SetSize(*section, Name(), "max_image_pixels", options.limits.max_image_pixels, 1);
    SetUInt32(*section, Name(), "max_image_width", options.limits.max_image_width, 1);
    SetUInt32(*section, Name(), "max_image_height", options.limits.max_image_height, 1);
    SetSize(*section, Name(), "max_prompt_bytes", options.limits.max_prompt_bytes, 1);
    SetInt32(*section, Name(), "min_context_size", options.limits.min_context_size, 1);
    SetInt32(*section, Name(), "max_context_size", options.limits.max_context_size, 1);
    SetInt32(*section, Name(), "max_vlm_tokens", options.limits.max_vlm_tokens, 1);
    SetFloat(*section, Name(), "max_temperature", options.limits.max_temperature, 0.0f, 100.0f);
    SetInt32(*section, Name(), "max_top_k", options.limits.max_top_k, 0);
    SetInt32(*section, Name(), "max_sequence_length", options.limits.max_sequence_length, 1);
    SetInt32(*section, Name(), "max_batch_size", options.limits.max_batch_size, 1);
    SetInt64(*section, Name(), "max_token_id", options.limits.max_token_id, 1);
    SetFloat(*section, Name(), "max_abs_personality", options.limits.max_abs_personality, 0.0f, 1000000.0f);
}

bool LimitsConfigSection::LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const {
    CliArgumentParser parser(cursor);
    CONFIG_VALUE_ARG(kMaxImageMb, value, options.limits.max_image_bytes = ParseMegabytesOption(kMaxImageMb, *value);)
    CONFIG_VALUE_ARG(kMaxImagePixels, value, options.limits.max_image_pixels = static_cast<std::size_t>(ParsePositiveOption(kMaxImagePixels, *value));)
    CONFIG_VALUE_ARG(kMaxImageWidth, value, options.limits.max_image_width = static_cast<std::uint32_t>(ParsePositiveOption(kMaxImageWidth, *value));)
    CONFIG_VALUE_ARG(kMaxImageHeight, value, options.limits.max_image_height = static_cast<std::uint32_t>(ParsePositiveOption(kMaxImageHeight, *value));)
    CONFIG_VALUE_ARG(kMaxPromptBytes, value, options.limits.max_prompt_bytes = static_cast<std::size_t>(ParsePositiveOption(kMaxPromptBytes, *value));)
    CONFIG_VALUE_ARG(kMaxSequenceLength, value, options.limits.max_sequence_length = ParsePositiveOption(kMaxSequenceLength, *value);)
    CONFIG_VALUE_ARG(kMaxBatchSize, value, options.limits.max_batch_size = ParsePositiveOption(kMaxBatchSize, *value);)
    CONFIG_VALUE_ARG(kMaxVlmTokens, value, options.limits.max_vlm_tokens = ParsePositiveOption(kMaxVlmTokens, *value);)
    CONFIG_VALUE_ARG(kMaxContextSize, value, options.limits.max_context_size = ParsePositiveOption(kMaxContextSize, *value);)
    CONFIG_VALUE_ARG(kMaxTokenId, value, options.limits.max_token_id = ParsePositiveOption(kMaxTokenId, *value);)
    return false;
}

void LimitsConfigSection::Validate(MultimodalServerOptions& options) const {
    if (options.limits.max_context_size < options.limits.min_context_size) {
        throw std::runtime_error("--max-context-size is below the minimum context size");
    }
}

} // namespace

REGISTER_CONFIG_SECTION(LimitsConfigSection)

} // namespace server_config
