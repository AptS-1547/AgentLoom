#pragma once

#include "result.h"
#include "thread_pool.h"

#include <cstddef>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace core {

struct KeyedSerialExecutorOptions {
    std::size_t max_keys = 256;
    std::size_t queue_capacity_per_key = 1024;
    std::string task_name_prefix = "keyed-serial";
};

class IKeyedSerialExecutor {
public:
    using Task = std::function<Status()>;

    virtual ~IKeyedSerialExecutor() = default;
    virtual Status Submit(std::string_view key, Task task, std::string task_name = {}) = 0;
    virtual Status WaitIdle(std::string_view key, std::chrono::milliseconds timeout) = 0;
};

class KeyedSerialExecutor final : public IKeyedSerialExecutor,
                                  public std::enable_shared_from_this<KeyedSerialExecutor> {
public:
    KeyedSerialExecutor(std::shared_ptr<ThreadPool> pool,
                        KeyedSerialExecutorOptions options = {});
    ~KeyedSerialExecutor() override;

    KeyedSerialExecutor(const KeyedSerialExecutor&) = delete;
    KeyedSerialExecutor& operator=(const KeyedSerialExecutor&) = delete;

    Status Submit(std::string_view key, Task task, std::string task_name = {}) override;
    Status WaitIdle(std::string_view key, std::chrono::milliseconds timeout) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace core
