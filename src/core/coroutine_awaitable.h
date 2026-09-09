#pragma once

// 回调式异步 / future 到 C++20 协程（co_await）的薄适配器。
//
// 目标：让下游（以及 AgentLoom 自身的新异步代码）在线性 co_await 写法里调用现有
// 回调式异步接口，不阻塞 worker 线程，也不改动 I* 接口契约。错误统一经
// core::Result<T> 传播，与现有回调接口保持一致。awater 是协程类型无关的
// （只依赖 <coroutine>），可被 asio::awaitable、自写 task 等任意 C++20 协程 co_await。

#include "result.h"

#include <coroutine>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>

namespace core::async {

// ── 回调式异步 → co_await ──────────────────────────────────────────────
// launch 形如 core::Status(Completion)，其中 Completion = std::function<void(core::Result<T>)>。
// 约定：launch 返回 Ok 表示已提交、completion 会在完成时恰好调用一次；
//       返回非 Ok 表示提交失败、completion 不会被 launch 调用（由 awaiter 补一次错误完成）。
// 同步完成 / 提交失败也正确处理：await_suspend 返回 false 不挂起、true 挂起。
template <typename T>
class CallbackAwaiter {
public:
    using Completion = std::function<void(core::Result<T>)>;

    template <typename Launch>
    explicit CallbackAwaiter(Launch launch) : launch_(std::move(launch)) {}

    CallbackAwaiter(const CallbackAwaiter&) = delete;
    CallbackAwaiter& operator=(const CallbackAwaiter&) = delete;
    // 回调式异步在进入处理流程前不知道结果，所以ready总是false，await_suspend里提交异步任务后挂起协程，等completion回调resume。
    bool await_ready() const noexcept { return false; }

    bool await_suspend(std::coroutine_handle<> handle) {
        {
            std::lock_guard lock(state_->mutex);
            state_->handle = handle;
            state_->awaiting = false;
        }
        // complete 按值传入 launch_（lambda 仅捕获 shared_ptr，拷贝廉价），
        // 失败路径保留一份用于补错误完成。
        auto complete = [state = state_](core::Result<T> result) {
            std::coroutine_handle<> resume;
            {
                std::lock_guard lock(state->mutex);
                state->result.emplace(std::move(result));
                if (state->awaiting) {
                    resume = state->handle;
                    state->handle = {};
                }
            }
            if (resume) resume.resume();
        };
        core::Status status = launch_(complete);
        if (!status.ok()) {
            complete(status);
        }
        std::lock_guard lock(state_->mutex);
        if (state_->result.has_value()) {
            return false;  // 同步完成或提交失败：不挂起。
        }
        state_->awaiting = true;
        return true;  // 挂起，等 completion 触发后 resume。
    }
    // resume 时，await_suspend 已经返回 false 或 completion 已经调用过 complete，取得结果。
    core::Result<T> await_resume() {
        std::lock_guard lock(state_->mutex);
        return std::move(*state_->result);
    }

private:
    struct State {
        std::mutex mutex;
        std::coroutine_handle<> handle{};
        std::optional<core::Result<T>> result;
        bool awaiting = false;
    };
    std::shared_ptr<State> state_ = std::make_shared<State>();
    std::function<core::Status(Completion)> launch_;
};

// 可取消回调操作到 co_await 的适配器。launch 返回持有底层操作生命周期的取消函数。
template <typename T>
class CancellableCallbackAwaiter {
public:
    using Completion = std::function<void(core::Result<T>)>;
    using Cancel = std::function<void()>;
    using Launch = std::function<core::Result<Cancel>(Completion)>;

    CancellableCallbackAwaiter(std::stop_token stop_token, Launch launch)
        : state_(std::make_shared<State>()),
          stop_token_(stop_token),
          launch_(std::move(launch)) {}

    CancellableCallbackAwaiter(const CancellableCallbackAwaiter&) = delete;
    CancellableCallbackAwaiter& operator=(const CancellableCallbackAwaiter&) = delete;

    bool await_ready() const noexcept { return false; }

    bool await_suspend(std::coroutine_handle<> handle) {
        {
            std::lock_guard lock(state_->mutex);
            state_->handle = handle;
        }
        auto complete = [state = state_](core::Result<T> result) {
            std::coroutine_handle<> resume;
            {
                std::lock_guard lock(state->mutex);
                if (state->result) return;
                state->result.emplace(std::move(result));
                state->cancel = {};
                if (state->awaiting) {
                    resume = state->handle;
                    state->handle = {};
                }
            }
            if (resume) resume.resume();
        };
        if (stop_token_.stop_requested()) {
            complete(core::Status::Error(
                core::ErrorCode::Cancelled,
                "callback operation cancelled before submission"));
        } else {
            auto submitted = launch_(complete);
            if (!submitted.ok()) {
                complete(submitted.status());
            } else {
                {
                    std::lock_guard lock(state_->mutex);
                    if (!state_->result) {
                        state_->cancel = std::move(submitted).value();
                    }
                }
                state_->stop_callback.emplace(
                    stop_token_,
                    [weak = std::weak_ptr<State>(state_)] {
                    auto state = weak.lock();
                    if (!state) return;
                    Cancel cancel;
                    {
                        std::lock_guard lock(state->mutex);
                        cancel = state->cancel;
                    }
                    if (cancel) cancel();
                });
            }
        }
        std::lock_guard lock(state_->mutex);
        if (state_->result) return false;
        state_->awaiting = true;
        return true;
    }

    core::Result<T> await_resume() {
        std::lock_guard lock(state_->mutex);
        return std::move(*state_->result);
    }

private:
    struct State {
        std::mutex mutex;
        std::coroutine_handle<> handle{};
        std::optional<core::Result<T>> result;
        Cancel cancel;
        bool awaiting = false;
        std::optional<std::stop_callback<std::function<void()>>> stop_callback;
    };

    std::shared_ptr<State> state_;
    std::stop_token stop_token_;
    Launch launch_;
};

// ── future → co_await ──────────────────────────────────────────────────
// 注意：std::future 不提供完成回调，非阻塞等待只能让一个专用线程 wait() 后 resume，
// 每个在途 future 占用一个线程。仅作迁移期兜底；长期应把 future 改回回调接口，
// 再用 CallbackAwaiter，避免「每 future 一线程」的开销。
template <typename T>
class FutureAwaiter {
public:
    explicit FutureAwaiter(std::future<T> future) : state_(std::make_shared<State>()) {
        state_->future = std::move(future);
    }

    FutureAwaiter(const FutureAwaiter&) = delete;
    FutureAwaiter& operator=(const FutureAwaiter&) = delete;

    bool await_ready() const {
        return state_->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    void await_suspend(std::coroutine_handle<> handle) {
        auto state = state_;
        std::thread([state, handle]() mutable {
            state->future.wait();
            handle.resume();
        }).detach();
    }

    T await_resume() {
        return state_->future.get();  // 已 ready，不阻塞。
    }

private:
    struct State {
        std::future<T> future;
    };
    std::shared_ptr<State> state_;
};

} // namespace core::async
