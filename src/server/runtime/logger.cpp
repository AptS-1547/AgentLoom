#include "logger.h"

#include "../../core/trace_context.h"

#include <iostream>
#include <memory>
#include <vector>

#include <spdlog/logger.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace logging {

namespace {

class TraceFlagFormatter final : public spdlog::custom_flag_formatter {
public:
    void format(const spdlog::details::log_msg&, const std::tm&,
                spdlog::memory_buf_t& dest) override {
        auto id = core::CurrentTraceId();
        dest.append(id.data(), id.data() + id.size());
    }

    std::unique_ptr<custom_flag_formatter> clone() const override {
        return std::make_unique<TraceFlagFormatter>();
    }
};

std::unique_ptr<spdlog::pattern_formatter> MakeFormatter(bool with_color) {
    auto formatter = std::make_unique<spdlog::pattern_formatter>();
    formatter->add_flag<TraceFlagFormatter>('*');
    if (with_color) {
        formatter->set_pattern("%Y-%m-%d %H:%M:%S.%e [%^%l%$] [%n] [%t] [%*] %v");
    } else {
        formatter->set_pattern("%Y-%m-%d %H:%M:%S.%e [%l] [%n] [%t] [%*] %v");
    }
    return formatter;
}

} // namespace

std::filesystem::path GetLogFilePath(const LoggerOptions& options) {
    return options.log_dir / options.file_name;
}

bool Initialize(const LoggerOptions& options) {
    try {
        std::filesystem::create_directories(options.log_dir);

        std::vector<spdlog::sink_ptr> sinks;
        if (options.enable_console) {
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_formatter(MakeFormatter(true));
            sinks.push_back(console_sink);
        }

        if (options.use_daily_rotation) {
            const auto log_file = GetLogFilePath(options).string();
            auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(
                log_file, options.daily_rotation_hour, options.daily_rotation_minute);
            file_sink->set_formatter(MakeFormatter(false));
            sinks.push_back(file_sink);
        } else {
            const auto log_file = GetLogFilePath(options).string();
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                log_file, options.max_file_size_bytes, options.max_files, true);
            file_sink->set_formatter(MakeFormatter(false));
            sinks.push_back(file_sink);
        }

        auto logger = std::make_shared<spdlog::logger>(
            options.logger_name, sinks.begin(), sinks.end());
        logger->set_level(spdlog::level::info);
        logger->flush_on(spdlog::level::info);

        spdlog::drop(options.logger_name);
        spdlog::register_logger(logger);
        spdlog::set_default_logger(logger);

        for (const auto& module_name : options.module_names) {
            auto module_dir = options.log_dir / module_name;
            std::filesystem::create_directories(module_dir);

            auto module_file = (module_dir / (module_name + ".log")).string();
            auto module_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(
                module_file, options.daily_rotation_hour, options.daily_rotation_minute);
            module_sink->set_formatter(MakeFormatter(false));

            std::vector<spdlog::sink_ptr> module_sinks;
            if (options.enable_console) {
                module_sinks.push_back(sinks.front());
            }
            module_sinks.push_back(module_sink);

            auto module_logger = std::make_shared<spdlog::logger>(
                module_name, module_sinks.begin(), module_sinks.end());
            module_logger->set_level(spdlog::level::info);
            module_logger->flush_on(spdlog::level::info);

            spdlog::drop(module_name);
            spdlog::register_logger(module_logger);
        }

        return true;
    } catch (const spdlog::spdlog_ex& e) {
        std::cerr << "[Logger] Failed to initialize spdlog: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[Logger] Failed to initialize logging: " << e.what() << std::endl;
        return false;
    }
}

std::shared_ptr<spdlog::logger> GetModuleLogger(const std::string& module_name) {
    auto logger = spdlog::get(module_name);
    if (logger) {
        return logger;
    }
    return spdlog::default_logger();
}

void Shutdown() {
    spdlog::shutdown();
}

} // namespace logging
