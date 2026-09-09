#include "../../src/core/coroutine_awaitable.h"
#include "../../src/core/coroutine_task.h"

#include <gtest/gtest.h>

#include <chrono>
#include <coroutine>
#include <future>
#include <optional>
#include <thread>

namespace {

// 极简协程类型：initial_suspend=never（创建即开始），final_suspend=always（完成后挂起，
// 由调用方检查 done() 后手动析构）。仅用于驱动测试，不含调度器。
template <typename T>
struct SimpleTask {
    struct promise_type {
        std::optional<T> value;
        std::exception_ptr error;
        SimpleTask get_return_object() {
            return SimpleTask{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_value(T v) { value = std::move(v); }
        void unhandled_exception() { error = std::current_exception(); }
    };

    std::coroutine_handle<promise_type> handle;
    explicit SimpleTask(std::coroutine_handle<promise_type> h) : handle(h) {}
    SimpleTask(const SimpleTask&) = delete;
    SimpleTask& operator=(const SimpleTask&) = delete;
    SimpleTask(SimpleTask&& other) noexcept : handle(std::exchange(other.handle, {})) {}
    ~SimpleTask() {
        if (handle) handle.destroy();
    }

    bool done() const { return !handle || handle.done(); }
    T result() {
        if (handle.promise().error) std::rethrow_exception(handle.promise().error);
        return std::move(*handle.promise().value);
    }
};

} // namespace

TEST(CoroutineAwaitableTest, SimpleTaskBaselineRunsToCompletion) {
    // 基线：无 co_await 的协程应立即完成，验证 SimpleTask 协程类型本身正确。
    SimpleTask<int> task = []() -> SimpleTask<int> {
        co_return 42;
    }();
    ASSERT_TRUE(task.done());
    EXPECT_EQ(task.result(), 42);
}

TEST(CoroutineAwaitableTest, CallbackAsyncCompletionResumesCoroutine) {
    // launch 提交到后台线程，稍后异步完成；协程应挂起后由回调 resume。
    std::promise<void> resumed;
    SimpleTask<core::Result<int>> task = [&resumed](int x) -> SimpleTask<core::Result<int>> {
        auto r = co_await core::async::CallbackAwaiter<int>([x](auto cb) {
            std::thread([x, cb = std::move(cb)]() mutable {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                cb(x * 2);
            }).detach();
            return core::Status::Ok();
        });
        resumed.set_value();
        co_return r;
    }(21);

    resumed.get_future().wait();
    ASSERT_TRUE(task.done());
    auto r = task.result();
    ASSERT_TRUE(r.ok()) << r.status().message();
    EXPECT_EQ(r.value(), 42);
}

TEST(CoroutineAwaitableTest, CallbackSyncCompletionDoesNotSuspend) {
    // launch 同步调用 cb，await_suspend 返回 true，协程不挂起。
    SimpleTask<core::Result<int>> task = []() -> SimpleTask<core::Result<int>> {
        auto r = co_await core::async::CallbackAwaiter<int>([](auto cb) {
            cb(7);
            return core::Status::Ok();
        });
        co_return r;
    }();

    ASSERT_TRUE(task.done());
    auto r = task.result();
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value(), 7);
}

TEST(CoroutineAwaitableTest, CallbackSubmitFailurePropagatesError) {
    // launch 返回非 Ok 且不调用 cb；awaiter 应补一次错误完成。
    SimpleTask<core::Result<int>> task = []() -> SimpleTask<core::Result<int>> {
        auto r = co_await core::async::CallbackAwaiter<int>([](auto) {
            return core::Status::Error(core::ErrorCode::Unavailable, "submit failed");
        });
        co_return r;
    }();

    ASSERT_TRUE(task.done());
    auto r = task.result();
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), core::ErrorCode::Unavailable);
}

TEST(CoroutineAwaitableTest, FutureAwaiterResumesWithoutBlockingCaller) {
    // future 由后台线程 resolve，协程经 FutureAwaiter 恢复，不阻塞调用线程。
    std::promise<int> promise;
    std::future<int> future = promise.get_future();
    SimpleTask<int> task = [](std::future<int> f) -> SimpleTask<int> {
        co_return co_await core::async::FutureAwaiter<int>(std::move(f));
    }(std::move(future));
    std::thread([&promise]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        promise.set_value(99);
    }).detach();

    // 轮询等待完成（测试用）；真实场景由 executor 驱动 resume。
    while (!task.done()) {
        std::this_thread::yield();
    }
    EXPECT_EQ(task.result(), 99);
}

TEST(CoroutineTaskTest, GetDrivesSyncTaskToCompletion) {
    core::async::task<core::Result<int>> t = []() -> core::async::task<core::Result<int>> {
        co_return 7;
    }();
    auto r = t.get();
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value(), 7);
}

TEST(CoroutineTaskTest, GetDrivesAsyncTaskToCompletion) {
    // task + co_await CallbackAwaiter + get()：回调在线程上触发，get() 阻塞等待完成。
    core::async::task<core::Result<int>> t = [](int x) -> core::async::task<core::Result<int>> {
        auto r = co_await core::async::CallbackAwaiter<int>([x](auto cb) {
            std::thread([x, cb = std::move(cb)]() mutable {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                cb(x * 3);
            }).detach();
            return core::Status::Ok();
        });
        co_return r;
    }(14);
    auto r = t.get();
    ASSERT_TRUE(r.ok()) << r.status().message();
    EXPECT_EQ(r.value(), 42);
}

TEST(CoroutineTaskTest, StartCompletesAsyncTaskWithoutBlockingCaller) {
    std::promise<core::Result<int>> completed;
    auto future = completed.get_future();
    core::async::task<int> task = []() -> core::async::task<int> {
        auto result = co_await core::async::CallbackAwaiter<int>([](auto callback) {
            std::thread([callback = std::move(callback)]() mutable {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                callback(42);
            }).detach();
            return core::Status::Ok();
        });
        if (!result.ok()) throw std::runtime_error(result.status().message());
        co_return std::move(result).value();
    }();

    std::move(task).Start([&completed](auto result) {
        completed.set_value(std::move(result));
    });

    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = future.get();
    ASSERT_TRUE(result.ok()) << result.status().message();
    EXPECT_EQ(result.value(), 42);
}

TEST(CoroutineTaskTest, CancellableCallbackAwaiterCancelsSubmittedOperation) {
    std::stop_source stop;
    std::promise<core::Result<int>> completed;
    auto future = completed.get_future();
    auto task = [&stop]() -> core::async::task<core::Result<int>> {
        co_return co_await core::async::CancellableCallbackAwaiter<int>(
            stop.get_token(),
            [](auto callback) -> core::Result<std::function<void()>> {
                return std::function<void()>([callback = std::move(callback)]() mutable {
                    callback(core::Status::Error(
                        core::ErrorCode::Cancelled,
                        "test operation cancelled"));
                });
            });
    }();
    std::move(task).Start([&completed](auto result) {
        if (!result.ok()) {
            completed.set_value(result.status());
            return;
        }
        completed.set_value(std::move(result).value());
    });

    stop.request_stop();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = future.get();
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::Cancelled);
}
