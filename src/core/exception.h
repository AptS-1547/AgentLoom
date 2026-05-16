#pragma once

#include "result.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace core {

class AppException : public std::runtime_error {
public:
    explicit AppException(Status status)
        : std::runtime_error(status.message()),
          status_(std::move(status)) {}

    explicit AppException(ErrorCode code, std::string message)
        : std::runtime_error(message),
          status_(code, std::move(message)) {}

    const Status& status() const noexcept {
        return status_;
    }

private:
    Status status_;
};

} // namespace core
