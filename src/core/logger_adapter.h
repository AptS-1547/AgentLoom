#pragma once

#include <spdlog/spdlog.h>

#include <memory>
#include <string_view>
#include <utility>

namespace core {

class LoggerAdapter {
public:
    LoggerAdapter() = default;
    explicit LoggerAdapter(std::shared_ptr<spdlog::logger> logger) noexcept
        : logger_(std::move(logger)) {}

    static LoggerAdapter ForModule(std::string_view module_name) {
        auto logger = spdlog::get(std::string(module_name));
        if (!logger) {
            logger = spdlog::default_logger();
        }
        return LoggerAdapter(std::move(logger));
    }

    bool valid() const noexcept {
        return static_cast<bool>(logger_);
    }

    spdlog::logger* get() const noexcept {
        return logger_.get();
    }

    template <typename... Args>
    void info(spdlog::format_string_t<Args...> fmt, Args&&... args) const {
        if (logger_) {
            logger_->info(fmt, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void warn(spdlog::format_string_t<Args...> fmt, Args&&... args) const {
        if (logger_) {
            logger_->warn(fmt, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void error(spdlog::format_string_t<Args...> fmt, Args&&... args) const {
        if (logger_) {
            logger_->error(fmt, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void debug(spdlog::format_string_t<Args...> fmt, Args&&... args) const {
        if (logger_) {
            logger_->debug(fmt, std::forward<Args>(args)...);
        }
    }

private:
    std::shared_ptr<spdlog::logger> logger_;
};

} // namespace core
