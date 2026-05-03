#include "logger.h"

#include <iostream>
#include <vector>

#include <spdlog/logger.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace logging {

std::filesystem::path GetLogFilePath(const LoggerOptions& options) {
    return options.log_dir / options.file_name;
}

bool Initialize(const LoggerOptions& options) {
    try {
        std::filesystem::create_directories(options.log_dir);

        std::vector<spdlog::sink_ptr> sinks;
        if (options.enable_console) {
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_pattern("%Y-%m-%d %H:%M:%S.%e [%^%l%$] [%t] %v");
            sinks.push_back(console_sink);
        }

        const auto log_file = GetLogFilePath(options).string();
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_file,
            options.max_file_size_bytes,
            options.max_files,
            true
        );
        file_sink->set_pattern("%Y-%m-%d %H:%M:%S.%e [%l] [%t] %v");
        sinks.push_back(file_sink);

        auto logger = std::make_shared<spdlog::logger>(
            options.logger_name,
            sinks.begin(),
            sinks.end()
        );
        logger->set_level(spdlog::level::info);
        logger->flush_on(spdlog::level::info);

        spdlog::drop(options.logger_name);
        spdlog::register_logger(logger);
        spdlog::set_default_logger(logger);
        return true;
    } catch (const spdlog::spdlog_ex& e) {
        std::cerr << "[Logger] Failed to initialize spdlog: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[Logger] Failed to initialize logging: " << e.what() << std::endl;
        return false;
    }
}

void Shutdown() {
    spdlog::shutdown();
}

} // namespace logging
