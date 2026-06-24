#include "config_section.h"

#include <stdexcept>

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(SkillSessionConfigSection, "skill_session")
    CONFIG_CLI_STRING(kEnabled, "--skill-session-enabled");
    CONFIG_CLI_STRING(kDisabled, "--skill-session-disabled");
    CONFIG_CLI_STRING(kStartupTimeout, "--skill-session-startup-timeout-ms");
    CONFIG_CLI_STRING(kMaxDuration, "--skill-session-max-duration-ms");
    CONFIG_CLI_STRING(kIdleTimeout, "--skill-session-idle-timeout-ms");
    CONFIG_CLI_STRING(kClosingTimeout, "--skill-session-closing-timeout-ms");
    CONFIG_CLI_STRING(kMaxRecentObservations, "--skill-session-max-recent-observations");
    CONFIG_CLI_STRING(kCleanupInterval, "--skill-session-cleanup-interval-seconds");
    void Validate(MultimodalServerOptions& options) const override;
};

void SkillSessionConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) {
        return;
    }
    SetBool(*section, Name(), "enabled", options.skill_session.enabled);
    SetInt(*section, Name(), "startup_timeout_ms", options.skill_session.startup_timeout_ms, 1);
    SetInt(*section, Name(), "max_duration_ms", options.skill_session.max_duration_ms, 1);
    SetInt(*section, Name(), "idle_timeout_ms", options.skill_session.idle_timeout_ms, 1);
    SetInt(*section, Name(), "closing_timeout_ms", options.skill_session.closing_timeout_ms, 1);
    SetSize(*section, Name(), "max_recent_observations", options.skill_session.max_recent_observations, 1);
    SetInt(*section, Name(), "cleanup_interval_seconds", options.skill_session.cleanup_interval_seconds, 1);
}

bool SkillSessionConfigSection::LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const {
    CliArgumentParser parser(cursor);

    CONFIG_FLAG_ARG(kEnabled, options.skill_session.enabled = true;)
    CONFIG_FLAG_ARG(kDisabled, options.skill_session.enabled = false;)
    CONFIG_VALUE_ARG(kStartupTimeout, value, options.skill_session.startup_timeout_ms = ParsePositiveOption(kStartupTimeout, *value);)
    CONFIG_VALUE_ARG(kMaxDuration, value, options.skill_session.max_duration_ms = ParsePositiveOption(kMaxDuration, *value);)
    CONFIG_VALUE_ARG(kIdleTimeout, value, options.skill_session.idle_timeout_ms = ParsePositiveOption(kIdleTimeout, *value);)
    CONFIG_VALUE_ARG(kClosingTimeout, value, options.skill_session.closing_timeout_ms = ParsePositiveOption(kClosingTimeout, *value);)
    CONFIG_VALUE_ARG(kMaxRecentObservations, value, options.skill_session.max_recent_observations = ParsePositiveOption(kMaxRecentObservations, *value);)
    CONFIG_VALUE_ARG(kCleanupInterval, value, options.skill_session.cleanup_interval_seconds = ParsePositiveOption(kCleanupInterval, *value);)

    return false;
}

void SkillSessionConfigSection::Validate(MultimodalServerOptions& options) const {
    const auto& skill = options.skill_session;
    if (skill.startup_timeout_ms <= 0) {
        throw std::runtime_error("skill_session.startup_timeout_ms must be positive");
    }
    if (skill.max_duration_ms <= 0) {
        throw std::runtime_error("skill_session.max_duration_ms must be positive");
    }
    if (skill.idle_timeout_ms <= 0) {
        throw std::runtime_error("skill_session.idle_timeout_ms must be positive");
    }
    if (skill.closing_timeout_ms <= 0) {
        throw std::runtime_error("skill_session.closing_timeout_ms must be positive");
    }
    if (skill.max_recent_observations == 0) {
        throw std::runtime_error("skill_session.max_recent_observations must be positive");
    }
    if (skill.cleanup_interval_seconds <= 0) {
        throw std::runtime_error("skill_session.cleanup_interval_seconds must be positive");
    }
}

} // namespace

REGISTER_CONFIG_SECTION(SkillSessionConfigSection)

} // namespace server_config
