#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace logging {

struct LoggerOptions {
    std::filesystem::path log_dir = "logs";
    std::string logger_name = "bert_inference_server";
    std::string file_name = "bert_inference_server.log";
    std::size_t max_file_size_bytes = 10 * 1024 * 1024;
    std::size_t max_files = 5;
    bool enable_console = true;
};

std::filesystem::path GetLogFilePath(const LoggerOptions& options);

bool Initialize(const LoggerOptions& options);

void Shutdown();

} // namespace logging
