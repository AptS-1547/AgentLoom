#pragma once

#include <optional>
#include <string>
#include <utility>

namespace core {

enum class ErrorCode {
    Ok = 0,
    Unknown,
    InvalidArgument,
    OutOfMemory,
    NotFound,
    Timeout,
    Cancelled,
    AlreadyExists,
    PermissionDenied,
    FailedPrecondition,
    Unimplemented,
    ResourceExhausted,
    Unavailable,
    InternalError
};

class Status {
public:
    Status() noexcept = default;
    Status(ErrorCode code, std::string message = {}) noexcept
        : code_(code), message_(std::move(message)) {}

    static Status Ok() noexcept {
        return Status();
    }

    static Status Error(ErrorCode code, std::string message = {}) noexcept {
        return Status(code, std::move(message));
    }

    bool ok() const noexcept {
        return code_ == ErrorCode::Ok;
    }

    explicit operator bool() const noexcept {
        return ok();
    }

    ErrorCode code() const noexcept {
        return code_;
    }

    const std::string& message() const noexcept {
        return message_;
    }

private:
    ErrorCode code_ = ErrorCode::Ok;
    std::string message_;
};

template <typename T>
class Result {
public:
    Result() = delete;

    Result(const T& value)
        : value_(value), status_(Status::Ok()) {}

    Result(T&& value)
        : value_(std::move(value)), status_(Status::Ok()) {}

    Result(const Status& status)
        : status_(status) {}

    Result(Status&& status)
        : status_(std::move(status)) {}

    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;
    Result(Result&&) noexcept = default;
    Result& operator=(Result&&) noexcept = default;
    ~Result() = default;

    bool has_value() const noexcept {
        return value_.has_value();
    }

    bool ok() const noexcept {
        return value_.has_value();
    }

    explicit operator bool() const noexcept {
        return value_.has_value();
    }

    const Status& status() const noexcept {
        return status_;
    }

    T& value() & {
        return value_.value();
    }

    const T& value() const & {
        return value_.value();
    }

    T&& value() && {
        return std::move(value_.value());
    }

private:
    std::optional<T> value_;
    Status status_;
};

template <>
class Result<void> {
public:
    Result() noexcept = default;
    Result(const Status& status) noexcept
        : status_(status) {}
    Result(Status&& status) noexcept
        : status_(std::move(status)) {}

    bool has_value() const noexcept {
        return status_.ok();
    }

    bool ok() const noexcept {
        return status_.ok();
    }

    explicit operator bool() const noexcept {
        return status_.ok();
    }

    const Status& status() const noexcept {
        return status_;
    }

private:
    Status status_ = Status::Ok();
};

} // namespace core
