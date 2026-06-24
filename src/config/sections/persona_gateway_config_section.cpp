#include "config_section.h"

#include <filesystem>
#include <stdexcept>

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(PersonaGatewayConfigSection, "persona_gateway")
    CONFIG_CLI_STRING(kWebSocketPath, "--gateway-ws-path");
    CONFIG_CLI_STRING(kStaticRoot, "--gateway-static-root");
    CONFIG_CLI_STRING(kStaticIndex, "--gateway-static-index");
    CONFIG_CLI_STRING(kStaticSpa, "--gateway-static-spa");
    CONFIG_CLI_STRING(kDocumentStoreEnabled, "--gateway-document-store-enabled");
    CONFIG_CLI_STRING(kDocumentStoreRoot, "--gateway-document-store-root");
    CONFIG_CLI_STRING(kDocumentStoreDb, "--gateway-document-store-db");
    CONFIG_CLI_STRING(kDocumentStoreRetentionHours, "--gateway-document-store-retention-hours");
    CONFIG_CLI_STRING(kDocumentStoreCleanupSeconds, "--gateway-document-store-cleanup-seconds");
    CONFIG_CLI_STRING(kComputeWorkers, "--gateway-compute-workers");
    CONFIG_CLI_STRING(kComputeQueue, "--gateway-compute-queue");
    CONFIG_CLI_STRING(kIoWorkers, "--gateway-io-workers");
    CONFIG_CLI_STRING(kIoQueue, "--gateway-io-queue");
    CONFIG_CLI_STRING(kSessionIdle, "--gateway-session-idle-minutes");
    CONFIG_CLI_STRING(kSessionTurns, "--gateway-session-max-recent-turns");
    CONFIG_CLI_STRING(kRecentRawTurns, "--gateway-runtime-recent-raw-turns");
    CONFIG_CLI_STRING(kDefaultModel, "--gateway-runtime-model");
    CONFIG_CLI_STRING(kFilterDisabled, "--gateway-filter-disabled");
    CONFIG_CLI_STRING(kFilterEnabled, "--gateway-filter-enabled");
    void Validate(MultimodalServerOptions& options) const override;
};

std::filesystem::path ResolveRelativeToConfig(const std::filesystem::path& path,
                                              const std::filesystem::path& config_path) {
    if (path.empty() || path.is_absolute() || config_path.empty()) {
        return path;
    }
    return config_path.parent_path() / path;
}

void LoadThreadPoolJson(const Json& section,
                        std::string_view section_name,
                        std::string_view field_name,
                        GatewayThreadPoolConfigOptions& options) {
    const Json* pool = FindField(section, section_name, field_name);
    if (!pool) {
        return;
    }
    if (!pool->is_object()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be an object");
    }
    SetSize(*pool, field_name, "worker_count", options.worker_count, 0);
    SetSize(*pool, field_name, "queue_capacity", options.queue_capacity, 0);
}

void PersonaGatewayConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) {
        return;
    }

    SetString(*section, Name(), "websocket_path", options.persona_gateway.websocket_path);
    SetString(*section, Name(), "address", options.http.address);
    if (const Json* port_field = FindField(*section, Name(), "port")) {
        if (port_field->is_number_unsigned()) {
            unsigned int port = port_field->get<unsigned int>();
            if (port > 65535) {
                throw std::runtime_error("persona_gateway.port must be <= 65535");
            }
            options.http.port = static_cast<unsigned short>(port);
        }
    }
    SetSize(*section, Name(), "http_io_threads", options.http.io_threads, 1);
    if (const Json* static_files = FindField(*section, Name(), "static_files")) {
        if (!static_files->is_object()) {
            throw std::runtime_error("persona_gateway.static_files must be an object");
        }
        SetBool(*static_files, "static_files", "enabled", options.persona_gateway.static_files.enabled);
        SetPath(*static_files, "static_files", "root", options.persona_gateway.static_files.root);
        SetString(*static_files, "static_files", "index_file", options.persona_gateway.static_files.index_file);
        SetBool(*static_files, "static_files", "spa_fallback", options.persona_gateway.static_files.spa_fallback);
    }
    if (const Json* document_store = FindField(*section, Name(), "document_store")) {
        if (!document_store->is_object()) {
            throw std::runtime_error("persona_gateway.document_store must be an object");
        }
        SetBool(*document_store, "document_store", "enabled", options.persona_gateway.document_store.enabled);
        SetPath(*document_store, "document_store", "root", options.persona_gateway.document_store.root);
        SetPath(*document_store, "document_store", "database_path", options.persona_gateway.document_store.database_path);
        SetSize(*document_store, "document_store", "read_connection_count", options.persona_gateway.document_store.read_connection_count, 1);
        SetSize(*document_store, "document_store", "write_connection_count", options.persona_gateway.document_store.write_connection_count, 1);
        SetInt(*document_store, "document_store", "busy_timeout_ms", options.persona_gateway.document_store.busy_timeout_ms, 1, 60000);
        SetInt(*document_store, "document_store", "retention_hours", options.persona_gateway.document_store.retention_hours, 1, 24 * 365);
        SetInt(*document_store, "document_store", "cleanup_interval_seconds", options.persona_gateway.document_store.cleanup_interval_seconds, 1, 24 * 3600);
    }
    LoadThreadPoolJson(*section, Name(), "compute_pool", options.persona_gateway.compute_pool);
    LoadThreadPoolJson(*section, Name(), "io_pool", options.persona_gateway.io_pool);
    SetInt(*section, Name(), "session_idle_timeout_minutes", options.persona_gateway.session_idle_timeout_minutes, 1, 1440);
    SetSize(*section, Name(), "session_max_recent_turns", options.persona_gateway.session_max_recent_turns, 1);
    SetSize(*section, Name(), "runtime_recent_raw_turns", options.persona_gateway.runtime_recent_raw_turns, 1);
    SetString(*section, Name(), "runtime_default_model", options.persona_gateway.runtime_default_model);
    if (const Json* filter = FindField(*section, Name(), "request_filter")) {
        if (!filter->is_object()) {
            throw std::runtime_error("persona_gateway.request_filter must be an object");
        }
        SetBool(*filter, "request_filter", "enabled", options.persona_gateway.request_filter_enabled);
        SetBool(*filter, "request_filter", "reject_control_chars", options.persona_gateway.reject_control_chars);
        SetBool(*filter, "request_filter", "reject_suspicious_patterns", options.persona_gateway.reject_suspicious_patterns);
    }
}

bool PersonaGatewayConfigSection::LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const {
    CliArgumentParser parser(cursor);

    CONFIG_VALUE_ARG(kWebSocketPath, value, options.persona_gateway.websocket_path = *value;)
    CONFIG_VALUE_ARG(kStaticRoot, value, {
        options.persona_gateway.static_files.enabled = true;
        options.persona_gateway.static_files.root = *value;
    })
    CONFIG_VALUE_ARG(kStaticIndex, value, options.persona_gateway.static_files.index_file = *value;)
    CONFIG_FLAG_ARG(kStaticSpa, {
        options.persona_gateway.static_files.enabled = true;
        options.persona_gateway.static_files.spa_fallback = true;
    })
    CONFIG_FLAG_ARG(kDocumentStoreEnabled, options.persona_gateway.document_store.enabled = true;)
    CONFIG_VALUE_ARG(kDocumentStoreRoot, value, {
        options.persona_gateway.document_store.enabled = true;
        options.persona_gateway.document_store.root = *value;
    })
    CONFIG_VALUE_ARG(kDocumentStoreDb, value, {
        options.persona_gateway.document_store.enabled = true;
        options.persona_gateway.document_store.database_path = *value;
    })
    CONFIG_VALUE_ARG(kDocumentStoreRetentionHours, value, {
        options.persona_gateway.document_store.enabled = true;
        options.persona_gateway.document_store.retention_hours = ParsePositiveOption(kDocumentStoreRetentionHours, *value);
    })
    CONFIG_VALUE_ARG(kDocumentStoreCleanupSeconds, value, {
        options.persona_gateway.document_store.enabled = true;
        options.persona_gateway.document_store.cleanup_interval_seconds = ParsePositiveOption(kDocumentStoreCleanupSeconds, *value);
    })
    CONFIG_VALUE_ARG(kComputeWorkers, value, options.persona_gateway.compute_pool.worker_count = ParseNonNegativeOption(kComputeWorkers, *value);)
    CONFIG_VALUE_ARG(kComputeQueue, value, options.persona_gateway.compute_pool.queue_capacity = ParseNonNegativeOption(kComputeQueue, *value);)
    CONFIG_VALUE_ARG(kIoWorkers, value, options.persona_gateway.io_pool.worker_count = ParseNonNegativeOption(kIoWorkers, *value);)
    CONFIG_VALUE_ARG(kIoQueue, value, options.persona_gateway.io_pool.queue_capacity = ParseNonNegativeOption(kIoQueue, *value);)
    CONFIG_VALUE_ARG(kSessionIdle, value, options.persona_gateway.session_idle_timeout_minutes = ParsePositiveOption(kSessionIdle, *value);)
    CONFIG_VALUE_ARG(kSessionTurns, value, options.persona_gateway.session_max_recent_turns = ParsePositiveOption(kSessionTurns, *value);)
    CONFIG_VALUE_ARG(kRecentRawTurns, value, options.persona_gateway.runtime_recent_raw_turns = ParsePositiveOption(kRecentRawTurns, *value);)
    CONFIG_VALUE_ARG(kDefaultModel, value, options.persona_gateway.runtime_default_model = *value;)
    CONFIG_FLAG_ARG(kFilterDisabled, options.persona_gateway.request_filter_enabled = false;)
    CONFIG_FLAG_ARG(kFilterEnabled, options.persona_gateway.request_filter_enabled = true;)

    return false;
}

void PersonaGatewayConfigSection::Validate(MultimodalServerOptions& options) const {
    auto& gateway = options.persona_gateway;
    if (gateway.websocket_path.empty() || gateway.websocket_path.front() != '/') {
        throw std::runtime_error("persona_gateway.websocket_path must start with '/'");
    }
    if (gateway.static_files.enabled) {
        if (gateway.static_files.root.empty()) {
            throw std::runtime_error("persona_gateway.static_files.root is required when static files are enabled");
        }
        gateway.static_files.root = ResolveRelativeToConfig(gateway.static_files.root, options.config_file_path);
        if (gateway.static_files.index_file.empty()) {
            throw std::runtime_error("persona_gateway.static_files.index_file must not be empty");
        }
    }
    if (gateway.document_store.enabled) {
        if (gateway.document_store.root.empty()) {
            throw std::runtime_error("persona_gateway.document_store.root is required when document store is enabled");
        }
        if (gateway.document_store.database_path.empty()) {
            throw std::runtime_error("persona_gateway.document_store.database_path is required when document store is enabled");
        }
        gateway.document_store.root = ResolveRelativeToConfig(gateway.document_store.root, options.config_file_path);
        gateway.document_store.database_path = ResolveRelativeToConfig(gateway.document_store.database_path, options.config_file_path);
    }
}

} // namespace

REGISTER_CONFIG_SECTION(PersonaGatewayConfigSection)

} // namespace server_config
