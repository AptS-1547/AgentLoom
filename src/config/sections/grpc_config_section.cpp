#include "config_section.h"

#include "server_common.h"

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(GrpcConfigSection, "grpc")
    CONFIG_CLI_STRING(kHost, "--host");
    CONFIG_CLI_STRING(kPort, "--port");
    CONFIG_CLI_STRING(kLogDir, "--log-dir");
    CONFIG_CLI_STRING(kNumCqs, "--grpc-num-cqs");
    CONFIG_CLI_STRING(kMinPollers, "--grpc-min-pollers");
    CONFIG_CLI_STRING(kMaxPollers, "--grpc-max-pollers");
    CONFIG_CLI_STRING(kMaxReceiveMb, "--max-recv-mb");
    CONFIG_CLI_STRING(kMaxSendMb, "--max-send-mb");
    CONFIG_CLI_STRING(kStatsLogInterval, "--stats-log-interval-seconds");
    CONFIG_CLI_STRING(kSlowRequestMs, "--slow-request-ms");
    void Validate(MultimodalServerOptions& options) const override;
};

void GrpcConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) {
        return;
    }
    SetString(*section, Name(), "host", options.grpc.host);
    SetString(*section, Name(), "port", options.grpc.port);
    SetString(*section, Name(), "log_dir", options.grpc.log_dir);
    SetInt(*section, Name(), "num_cqs", options.grpc.grpc_num_cqs, 0);
    SetInt(*section, Name(), "min_pollers", options.grpc.grpc_min_pollers, 0);
    SetInt(*section, Name(), "max_pollers", options.grpc.grpc_max_pollers, 0);
    SetInt(*section, Name(), "max_receive_message_mb", options.grpc.max_receive_message_mb, 1);
    SetInt(*section, Name(), "max_send_message_mb", options.grpc.max_send_message_mb, 1);
    SetInt(*section, Name(), "stats_log_interval_seconds", options.grpc.stats_log_interval_seconds, 0);
    SetInt(*section, Name(), "slow_request_ms", options.grpc.slow_request_ms, 0);
}

bool GrpcConfigSection::LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const {
    CliArgumentParser parser(cursor);
    CONFIG_VALUE_ARG(kHost, value, options.grpc.host = *value;)
    CONFIG_VALUE_ARG(kPort, value, options.grpc.port = *value;)
    CONFIG_VALUE_ARG(kLogDir, value, options.grpc.log_dir = *value;)
    CONFIG_VALUE_ARG(kNumCqs, value, options.grpc.grpc_num_cqs = ParseCliInt(kNumCqs, *value);)
    CONFIG_VALUE_ARG(kMinPollers, value, options.grpc.grpc_min_pollers = ParseCliInt(kMinPollers, *value);)
    CONFIG_VALUE_ARG(kMaxPollers, value, options.grpc.grpc_max_pollers = ParseCliInt(kMaxPollers, *value);)
    CONFIG_VALUE_ARG(kMaxReceiveMb, value, options.grpc.max_receive_message_mb = ParseCliInt(kMaxReceiveMb, *value);)
    CONFIG_VALUE_ARG(kMaxSendMb, value, options.grpc.max_send_message_mb = ParseCliInt(kMaxSendMb, *value);)
    CONFIG_VALUE_ARG(kStatsLogInterval, value, options.grpc.stats_log_interval_seconds = ParseCliInt(kStatsLogInterval, *value);)
    CONFIG_VALUE_ARG(kSlowRequestMs, value, options.grpc.slow_request_ms = ParseCliInt(kSlowRequestMs, *value);)
    return false;
}

void GrpcConfigSection::Validate(MultimodalServerOptions& options) const {
    if (options.grpc.grpc_num_cqs <= 0) {
        options.grpc.grpc_num_cqs = server_common::ResolveDefaultGrpcNumCqs();
    }
    if (options.grpc.grpc_min_pollers <= 0) {
        options.grpc.grpc_min_pollers = 1;
    }
    if (options.grpc.grpc_max_pollers <= 0) {
        options.grpc.grpc_max_pollers = server_common::ResolveDefaultGrpcMaxPollers();
    }
    if (options.grpc.grpc_max_pollers < options.grpc.grpc_min_pollers) {
        options.grpc.grpc_max_pollers = options.grpc.grpc_min_pollers;
    }
    if (options.grpc.stats_log_interval_seconds < 0) {
        options.grpc.stats_log_interval_seconds = 0;
    }
    if (options.grpc.slow_request_ms < 0) {
        options.grpc.slow_request_ms = 0;
    }
}

} // namespace

REGISTER_CONFIG_SECTION(GrpcConfigSection)

} // namespace server_config
