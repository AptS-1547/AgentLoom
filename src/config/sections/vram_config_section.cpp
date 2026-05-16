#include "config_section.h"

#include <stdexcept>

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(VramGuardConfigSection, "vram_guard")
    CONFIG_CLI_STRING(kMonitorInterval, "--vram-monitor-interval-seconds");
    CONFIG_CLI_STRING(kWarningFreeMb, "--vram-warning-free-mb");
    CONFIG_CLI_STRING(kUnloadFreeMb, "--vram-unload-free-mb");
    CONFIG_CLI_STRING(kMinFreeBeforeLoadMb, "--vram-min-free-before-load-mb");
    CONFIG_CLI_STRING(kReloadAfterUnload, "--vram-reload-after-unload");
    CONFIG_CLI_STRING(kNoUnloadOnOom, "--no-vram-unload-on-oom-error");
    void Validate(MultimodalServerOptions& options) const override;
};

void VramGuardConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) {
        return;
    }
    SetInt(*section, Name(), "monitor_interval_seconds", options.vram.monitor_interval_seconds, 0);
    SetMegabytes(*section, Name(), "warning_free_mb", options.vram.warning_free_bytes);
    SetMegabytes(*section, Name(), "unload_free_mb", options.vram.unload_free_bytes);
    SetMegabytes(*section, Name(), "min_free_before_load_mb", options.vram.min_free_before_load_bytes);
    SetBool(*section, Name(), "reload_after_unload", options.vram.reload_after_unload);
    SetBool(*section, Name(), "unload_on_oom_error", options.vram.unload_on_oom_error);
}

bool VramGuardConfigSection::LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const {
    CliArgumentParser parser(cursor);
    CONFIG_VALUE_ARG(kMonitorInterval, value, options.vram.monitor_interval_seconds = ParseNonNegativeOption(kMonitorInterval, *value);)
    CONFIG_VALUE_ARG(kWarningFreeMb, value, options.vram.warning_free_bytes = ParseOptionalMegabytesOption(kWarningFreeMb, *value);)
    CONFIG_VALUE_ARG(kUnloadFreeMb, value, options.vram.unload_free_bytes = ParseOptionalMegabytesOption(kUnloadFreeMb, *value);)
    CONFIG_VALUE_ARG(kMinFreeBeforeLoadMb, value, options.vram.min_free_before_load_bytes = ParseOptionalMegabytesOption(kMinFreeBeforeLoadMb, *value);)
    CONFIG_FLAG_ARG(kReloadAfterUnload, options.vram.reload_after_unload = true;)
    CONFIG_FLAG_ARG(kNoUnloadOnOom, options.vram.unload_on_oom_error = false;)
    return false;
}

void VramGuardConfigSection::Validate(MultimodalServerOptions& options) const {
    if (options.vram.warning_free_bytes > 0 &&
        options.vram.unload_free_bytes > 0 &&
        options.vram.unload_free_bytes > options.vram.warning_free_bytes) {
        throw std::runtime_error("--vram-unload-free-mb must be less than or equal to --vram-warning-free-mb");
    }
}

} // namespace

REGISTER_CONFIG_SECTION(VramGuardConfigSection)

} // namespace server_config
