#pragma once

#include <optional>
#include <string>
#include <utility>
#include <type_traits>

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

/// 跨模块统一状态；成功状态的 code 为 ErrorCode::Ok。
class Status {
public:
    Status() noexcept = default;
    Status(ErrorCode code, std::string message = {}) noexcept
        : code_(code), message_(std::move(message)) {}

    static Status Ok() noexcept {
        return Status();
    }

    /// 构造失败状态。
    /// @param code 错误分类，不应传入 ErrorCode::Ok。
    /// @param message 面向日志和诊断的安全错误摘要，不应包含凭据。
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
/// 携带成功值或失败 Status 的 move-only 返回类型。
class Result {
public:
    Result() = delete;

    // SFINAE: disable value constructors when T=Status to avoid ambiguity
    template <typename U = T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<U>, Status>>>
    Result(const T& value)
        : value_(value), status_(Status::Ok()) {}

    template <typename U = T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<U>, Status>>>
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

    /// 返回状态；成功结果返回 Status::Ok()。
    const Status& status() const noexcept {
        return status_;
    }

    /// 取得成功值；调用前必须确认 ok()，失败时会抛出 bad_optional_access。
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
/// 无返回值操作的 Result 特化。
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
