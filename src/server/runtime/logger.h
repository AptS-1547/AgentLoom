#pragma once

#include <spdlog/spdlog.h>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#define LOG_INFO(...)    spdlog::info(__VA_ARGS__)
#define LOG_WARN(...)    spdlog::warn(__VA_ARGS__)
#define LOG_ERROR(...)   spdlog::error(__VA_ARGS__)
#define LOG_DEBUG(...)   spdlog::debug(__VA_ARGS__)

namespace logging {

struct LoggerOptions {
    std::filesystem::path log_dir = "logs";
    std::string logger_name = "bert_inference_server";
    std::string file_name = "bert_inference_server.log";
    std::size_t max_file_size_bytes = 10 * 1024 * 1024;
    std::size_t max_files = 5;
    bool enable_console = true;
    bool use_daily_rotation = true;
    int daily_rotation_hour = 0;
    int daily_rotation_minute = 0;
    std::vector<std::string> module_names;
};

std::filesystem::path GetLogFilePath(const LoggerOptions& options);

bool Initialize(const LoggerOptions& options);

std::shared_ptr<spdlog::logger> GetModuleLogger(const std::string& module_name);

void Shutdown();

} // namespace logging
