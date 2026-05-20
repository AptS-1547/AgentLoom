#include "config_section.h"

#include <stdexcept>

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(HttpConfigSection, "http")
    CONFIG_CLI_STRING(kAddress, "--http-address");
    CONFIG_CLI_STRING(kPort, "--http-port");
    CONFIG_CLI_STRING(kThreads, "--http-threads");
    CONFIG_CLI_STRING(kRequestTimeout, "--http-request-timeout");
    CONFIG_CLI_STRING(kWebSocketTimeout, "--http-websocket-timeout");
    CONFIG_CLI_STRING(kBodyLimit, "--http-body-limit");
    CONFIG_CLI_STRING(kWebSocketBufferLimit, "--http-websocket-buffer-limit");
};

void HttpConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) {
        return;
    }

    SetString(*section, Name(), "address", options.http.address);

    if (const Json* port_field = FindField(*section, Name(), "port")) {
        if (port_field->is_number_unsigned()) {
            unsigned int port = port_field->get<unsigned int>();
            if (port > 65535) {
                throw std::runtime_error("http.port must be <= 65535");
            }
            options.http.port = static_cast<unsigned short>(port);
        }
    }

    SetSize(*section, Name(), "io_threads", options.http.io_threads, 1);

    if (const Json* timeout_field = FindField(*section, Name(), "request_timeout_seconds")) {
        if (timeout_field->is_number_unsigned()) {
            options.http.request_timeout = std::chrono::seconds(timeout_field->get<unsigned int>());
        }
    }

    if (const Json* ws_timeout_field = FindField(*section, Name(), "websocket_idle_timeout_seconds")) {
        if (ws_timeout_field->is_number_unsigned()) {
            options.http.websocket_idle_timeout = std::chrono::seconds(ws_timeout_field->get<unsigned int>());
        }
    }

    SetMegabytes(*section, Name(), "request_body_limit_mb", options.http.request_body_limit, 1);
    SetMegabytes(*section, Name(), "websocket_read_buffer_limit_mb", options.http.websocket_read_buffer_limit, 1);
}

bool HttpConfigSection::LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const {
    CliArgumentParser parser(cursor);

    CONFIG_VALUE_ARG(kAddress, value, options.http.address = *value;)
    CONFIG_VALUE_ARG(kPort, value, {
        int port = ParseCliInt(kPort, *value);
        if (port < 1 || port > 65535) {
            throw std::runtime_error("--http-port must be 1-65535");
        }
        options.http.port = static_cast<unsigned short>(port);
    })
    CONFIG_VALUE_ARG(kThreads, value, {
        options.http.io_threads = ParsePositiveOption(kThreads, *value);
    })
    CONFIG_VALUE_ARG(kRequestTimeout, value, {
        int seconds = ParseNonNegativeOption(kRequestTimeout, *value);
        options.http.request_timeout = std::chrono::seconds(seconds);
    })
    CONFIG_VALUE_ARG(kWebSocketTimeout, value, {
        int seconds = ParseNonNegativeOption(kWebSocketTimeout, *value);
        options.http.websocket_idle_timeout = std::chrono::seconds(seconds);
    })
    CONFIG_VALUE_ARG(kBodyLimit, value, {
        options.http.request_body_limit = ParseMegabytesOption(kBodyLimit, *value);
    })
    CONFIG_VALUE_ARG(kWebSocketBufferLimit, value, {
        options.http.websocket_read_buffer_limit = ParseMegabytesOption(kWebSocketBufferLimit, *value);
    })

    return false;
}

} // namespace

REGISTER_CONFIG_SECTION(HttpConfigSection)

} // namespace server_config
