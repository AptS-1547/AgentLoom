#pragma once

#include "result.h"

// 极简惰性协程任务：co_await 时才启动，可被 co_await 获取结果，无内建调度器。
// 挂起/恢复由被 co_await 的 awaiter（如 core::async::CallbackAwaiter）驱动；
// 对称转移（symmetric transfer）避免深层协程链导致的调用栈增长。
// get() 提供从同步上下文驱动任务到完成的阻塞入口（条件变量等待，非忙等）。

#include <coroutine>
#include <exception>
#include <functional>
#include <future>
#include <optional>
#include <utility>

namespace core::async {

template <typename T>
class task {
public:
    struct promise_type {
        std::optional<T> result;
        std::exception_ptr error;
        std::coroutine_handle<> continuation;
        std::function<void()> on_complete;  // final_suspend 时调用（get() 用于通知完成）
        std::function<void(core::Result<T>)> detached_completion;

        task get_return_object() noexcept {
            return task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        auto final_suspend() noexcept {
            struct FinalAwaiter {
                bool await_ready() const noexcept { return false; }
                std::coroutine_handle<> await_suspend(
                    std::coroutine_handle<promise_type> handle) noexcept {
                    auto& promise = handle.promise();
                    if (promise.on_complete) {
                        auto callback = std::move(promise.on_complete);
                        try {
                            callback();
                        } catch (...) {
                        }
                    }
                    if (promise.detached_completion) {
                        auto completion = std::move(promise.detached_completion);
                        core::Result<T> result = promise.error
                            ? ExceptionResult(promise.error)
                            : core::Result<T>(std::move(*promise.result));
                        try {
                            completion(std::move(result));
                        } catch (...) {
                        }
                        handle.destroy();
                        return std::noop_coroutine();
                    }
                    const auto continuation = promise.continuation;
                    if (continuation) return continuation;  // 对称转移给等待者
                    return std::noop_coroutine();
                }
                void await_resume() const noexcept {}
            };
            return FinalAwaiter{};
        }
        void return_value(T value) { result = std::move(value); }
        void unhandled_exception() { error = std::current_exception(); }

    private:
        static core::Result<T> ExceptionResult(const std::exception_ptr& error) {
            try {
                std::rethrow_exception(error);
            } catch (const std::exception& exception) {
                return core::Status::Error(core::ErrorCode::InternalError, exception.what());
            } catch (...) {
                return core::Status::Error(
                    core::ErrorCode::InternalError,
                    "coroutine task failed with an unknown exception");
            }
        }
    };

    using handle_type = std::coroutine_handle<promise_type>;

    task() noexcept : handle_(nullptr) {}
    explicit task(handle_type handle) noexcept : handle_(handle) {}
    task(task&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    task& operator=(task&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }
    task(const task&) = delete;
    task& operator=(const task&) = delete;
    ~task() {
        if (handle_) handle_.destroy();
    }

    // task 本身可被 co_await：co_await 时经对称转移启动并等待完成。
    bool await_ready() const noexcept { return false; }
    std::coroutine_handle<> await_suspend(std::coroutine_handle<> continuation) noexcept {
        handle_.promise().continuation = continuation;
        return handle_;
    }
    T await_resume() {
        if (handle_.promise().error) std::rethrow_exception(handle_.promise().error);
        return std::move(*handle_.promise().result);
    }

    // 从同步上下文驱动任务到完成（阻塞等待，非忙等）。
    T get() {
        std::promise<T> promise;
        auto future = promise.get_future();
        handle_.promise().on_complete = [this, &promise]() {
            if (handle_.promise().error) {
                promise.set_exception(handle_.promise().error);
            } else {
                promise.set_value(std::move(*handle_.promise().result));
            }
        };
        handle_.resume();  // 启动惰性任务
        return future.get();
    }

    // 从 callback/reactor 上下文非阻塞启动任务；完成后自动销毁协程帧。
    void Start(std::function<void(core::Result<T>)> completion) && {
        if (!handle_) {
            completion(core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "coroutine task has no state"));
            return;
        }
        auto handle = std::exchange(handle_, nullptr);
        handle.promise().detached_completion = std::move(completion);
        handle.resume();
    }

private:
    handle_type handle_;
};

} // namespace core::async
