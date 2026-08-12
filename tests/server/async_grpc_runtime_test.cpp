#include "async_emotion_grpc_service.h"

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <stdexcept>
#include <utility>

namespace {

using namespace std::chrono_literals;
using EmotionHandler = grpc_runtime::IAsyncUnaryRpcHandler<
    multimodal_inference::EmotionRequest,
    multimodal_inference::EmotionResponse>;

class GrpcServerHarness {
public:
    explicit GrpcServerHarness(grpc::Service& service) {
        grpc::ServerBuilder builder;
        int selected_port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
        builder.RegisterService(&service);
        server_ = builder.BuildAndStart();
        if (!server_ || selected_port <= 0) {
            throw std::runtime_error("failed to start async gRPC test server");
        }
        channel_ = grpc::CreateChannel(
            "127.0.0.1:" + std::to_string(selected_port),
            grpc::InsecureChannelCredentials());
    }

    ~GrpcServerHarness() {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
    }

    std::shared_ptr<grpc::Channel> channel() const { return channel_; }

private:
    std::unique_ptr<grpc::Server> server_;
    std::shared_ptr<grpc::Channel> channel_;
};

class LambdaEmotionHandler final : public EmotionHandler {
public:
    using Callback = std::function<core::Status(
        const grpc_runtime::AsyncGrpcCallContext&,
        const multimodal_inference::EmotionRequest&,
        multimodal_inference::EmotionResponse&)>;

    explicit LambdaEmotionHandler(Callback callback)
        : callback_(std::move(callback)) {}

    core::Status Handle(const grpc_runtime::AsyncGrpcCallContext& context,
                        const multimodal_inference::EmotionRequest& request,
                        multimodal_inference::EmotionResponse& response) override {
        return callback_(context, request, response);
    }

private:
    Callback callback_;
};

multimodal_inference::EmotionRequest Request() {
    multimodal_inference::EmotionRequest request;
    request.add_input_ids(1);
    return request;
}

std::shared_ptr<core::ThreadPool> StartedPool(std::size_t workers = 1) {
    core::ThreadPoolOptions options;
    options.worker_count = workers;
    options.name = "async-grpc-runtime-test";
    auto pool = std::make_shared<core::ThreadPool>(std::move(options));
    EXPECT_TRUE(pool->Start().ok());
    return pool;
}

std::shared_ptr<grpc_runtime::AsyncGrpcRuntime> CreateRuntime(
    std::shared_ptr<core::ThreadPool> pool,
    grpc_runtime::AsyncGrpcRuntimeOptions options = {}) {
    auto runtime = grpc_runtime::AsyncGrpcRuntime::Create(std::move(pool), std::move(options));
    EXPECT_TRUE(runtime.ok()) << runtime.status().message();
    return std::move(runtime).value();
}

TEST(AsyncGrpcRuntimeTest, RejectsInvalidRuntimeConfiguration) {
    auto missing_pool = grpc_runtime::AsyncGrpcRuntime::Create(nullptr);
    EXPECT_EQ(missing_pool.status().code(), core::ErrorCode::InvalidArgument);

    grpc_runtime::AsyncGrpcRuntimeOptions options;
    options.max_inflight_calls = 0;
    auto zero_capacity = grpc_runtime::AsyncGrpcRuntime::Create(StartedPool(), options);
    EXPECT_EQ(zero_capacity.status().code(), core::ErrorCode::InvalidArgument);
}

TEST(AsyncGrpcRuntimeTest, PropagatesTraceDeadlineAndResponse) {
    auto pool = StartedPool();
    auto runtime = CreateRuntime(pool);
    std::optional<grpc_runtime::AsyncGrpcCallContext> observed_context;
    auto handler = std::make_shared<LambdaEmotionHandler>(
        [&observed_context](const auto& context, const auto&, auto& response) {
            observed_context = context;
            response.add_emotion_logits(0.75f);
            return core::Status::Ok();
        });
    server::grpc_service::AsyncEmotionGrpcService service(
        runtime, handler);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext context;
    context.AddMetadata("x-trace-id", "async-trace-1");
    context.set_deadline(std::chrono::system_clock::now() + 5s);
    multimodal_inference::EmotionResponse response;
    const auto status = stub->PredictEmotion(&context, Request(), &response);

    ASSERT_TRUE(status.ok()) << status.error_message();
    ASSERT_TRUE(observed_context.has_value());
    EXPECT_EQ(observed_context->trace_id, "async-trace-1");
    EXPECT_EQ(observed_context->method_name, "PredictEmotion");
    EXPECT_GT(observed_context->deadline, std::chrono::system_clock::now());
    ASSERT_EQ(response.emotion_logits_size(), 1);
    EXPECT_FLOAT_EQ(response.emotion_logits(0), 0.75f);
    pool->Shutdown(true);
}

TEST(AsyncGrpcRuntimeTest, RejectsWhenInflightLimitReached) {
    auto pool = StartedPool();
    grpc_runtime::AsyncGrpcRuntimeOptions options;
    options.max_inflight_calls = 1;
    auto runtime = CreateRuntime(pool, options);
    std::mutex mutex;
    std::condition_variable entered_cv;
    std::condition_variable release_cv;
    bool entered = false;
    bool release = false;
    auto handler = std::make_shared<LambdaEmotionHandler>(
        [&](const auto&, const auto&, auto&) {
            std::unique_lock lock(mutex);
            entered = true;
            entered_cv.notify_all();
            release_cv.wait(lock, [&] { return release; });
            return core::Status::Ok();
        });
    server::grpc_service::AsyncEmotionGrpcService service(runtime, handler);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext first_context;
    multimodal_inference::EmotionResponse first_response;
    grpc::Status first_status;
    std::thread first_client([&] {
        first_status = stub->PredictEmotion(&first_context, Request(), &first_response);
    });

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(entered_cv.wait_for(lock, 2s, [&] { return entered; }));
    }

    grpc::ClientContext second_context;
    multimodal_inference::EmotionResponse second_response;
    const auto second_status = stub->PredictEmotion(&second_context, Request(), &second_response);
    EXPECT_EQ(second_status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);

    {
        std::lock_guard lock(mutex);
        release = true;
    }
    release_cv.notify_all();
    first_client.join();
    EXPECT_TRUE(first_status.ok()) << first_status.error_message();

    const auto snapshot = runtime->Snapshot();
    EXPECT_EQ(snapshot.rejected_calls, 1);
    pool->Shutdown(true);
}

TEST(AsyncGrpcRuntimeTest, CancellationRequestsStopToken) {
    auto pool = StartedPool();
    auto runtime = CreateRuntime(pool);
    std::mutex mutex;
    std::condition_variable entered_cv;
    std::condition_variable cancelled_cv;
    bool entered = false;
    bool cancelled = false;
    auto handler = std::make_shared<LambdaEmotionHandler>(
        [&](const auto& context, const auto&, auto&) {
            {
                std::lock_guard lock(mutex);
                entered = true;
            }
            entered_cv.notify_all();
            while (!context.stop_token.stop_requested()) {
                std::this_thread::sleep_for(2ms);
            }
            {
                std::lock_guard lock(mutex);
                cancelled = true;
            }
            cancelled_cv.notify_all();
            return core::Status::Error(core::ErrorCode::Cancelled, "client cancelled");
        });
    server::grpc_service::AsyncEmotionGrpcService service(runtime, handler);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext context;
    multimodal_inference::EmotionResponse response;
    grpc::Status status;
    std::thread client([&] {
        status = stub->PredictEmotion(&context, Request(), &response);
    });
    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(entered_cv.wait_for(lock, 2s, [&] { return entered; }));
    }
    context.TryCancel();
    {
        std::unique_lock lock(mutex);
        EXPECT_TRUE(cancelled_cv.wait_for(lock, 2s, [&] { return cancelled; }));
    }
    client.join();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    pool->Shutdown(true);
}

TEST(AsyncGrpcRuntimeTest, CancelsQueuedCallBeforeWorkerStarts) {
    auto pool = StartedPool();
    grpc_runtime::AsyncGrpcRuntimeOptions options;
    options.max_inflight_calls = 2;
    auto runtime = CreateRuntime(pool, options);
    std::mutex mutex;
    std::condition_variable entered_cv;
    std::condition_variable release_cv;
    bool first_entered = false;
    bool release_first = false;
    std::atomic<int> handler_calls{0};
    auto handler = std::make_shared<LambdaEmotionHandler>(
        [&](const auto&, const auto&, auto&) {
            const int call_index = handler_calls.fetch_add(1, std::memory_order_relaxed);
            if (call_index != 0) {
                return core::Status::Ok();
            }
            std::unique_lock lock(mutex);
            first_entered = true;
            entered_cv.notify_all();
            release_cv.wait(lock, [&] { return release_first; });
            return core::Status::Ok();
        });
    server::grpc_service::AsyncEmotionGrpcService service(runtime, handler);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext first_context;
    multimodal_inference::EmotionResponse first_response;
    grpc::Status first_status;
    std::thread first_client([&] {
        first_status = stub->PredictEmotion(&first_context, Request(), &first_response);
    });
    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(entered_cv.wait_for(lock, 2s, [&] { return first_entered; }));
    }

    grpc::ClientContext queued_context;
    multimodal_inference::EmotionResponse queued_response;
    grpc::Status queued_status;
    std::thread queued_client([&] {
        queued_status = stub->PredictEmotion(&queued_context, Request(), &queued_response);
    });
    const auto queued_deadline = std::chrono::steady_clock::now() + 2s;
    while (runtime->Snapshot().inflight_calls != 2 &&
           std::chrono::steady_clock::now() < queued_deadline) {
        std::this_thread::sleep_for(2ms);
    }
    ASSERT_EQ(runtime->Snapshot().inflight_calls, 2u);
    queued_context.TryCancel();

    const auto cancellation_deadline = std::chrono::steady_clock::now() + 2s;
    auto cancellation_snapshot = runtime->Snapshot();
    while (cancellation_snapshot.cancelled_calls != 1 &&
           std::chrono::steady_clock::now() < cancellation_deadline) {
        std::this_thread::sleep_for(2ms);
        cancellation_snapshot = runtime->Snapshot();
    }
    ASSERT_EQ(cancellation_snapshot.cancelled_calls, 1u);
    queued_client.join();
    EXPECT_EQ(queued_status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(handler_calls.load(std::memory_order_relaxed), 1);

    {
        std::lock_guard lock(mutex);
        release_first = true;
    }
    release_cv.notify_all();
    first_client.join();
    EXPECT_TRUE(first_status.ok()) << first_status.error_message();

    const auto completion_deadline = std::chrono::steady_clock::now() + 2s;
    auto snapshot = runtime->Snapshot();
    while ((snapshot.completed_calls != 2 || snapshot.inflight_calls != 0) &&
           std::chrono::steady_clock::now() < completion_deadline) {
        std::this_thread::sleep_for(2ms);
        snapshot = runtime->Snapshot();
    }
    EXPECT_EQ(snapshot.inflight_calls, 0);
    EXPECT_EQ(snapshot.cancelled_calls, 1);
    EXPECT_EQ(snapshot.completed_calls, 2);
    pool->Shutdown(true);
}

TEST(AsyncGrpcRuntimeTest, FinishesQueuedCallWhenThreadPoolDiscardsIt) {
    auto pool = StartedPool();
    grpc_runtime::AsyncGrpcRuntimeOptions options;
    options.max_inflight_calls = 2;
    auto runtime = CreateRuntime(pool, options);
    std::mutex mutex;
    std::condition_variable entered_cv;
    std::condition_variable release_cv;
    bool first_entered = false;
    bool release_first = false;
    std::atomic<int> handler_calls{0};
    auto handler = std::make_shared<LambdaEmotionHandler>(
        [&](const auto&, const auto&, auto&) {
            const int call_index = handler_calls.fetch_add(1, std::memory_order_relaxed);
            if (call_index != 0) {
                return core::Status::Ok();
            }
            std::unique_lock lock(mutex);
            first_entered = true;
            entered_cv.notify_all();
            release_cv.wait(lock, [&] { return release_first; });
            return core::Status::Ok();
        });
    server::grpc_service::AsyncEmotionGrpcService service(runtime, handler);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext first_context;
    multimodal_inference::EmotionResponse first_response;
    grpc::Status first_status;
    std::thread first_client([&] {
        first_status = stub->PredictEmotion(&first_context, Request(), &first_response);
    });
    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(entered_cv.wait_for(lock, 2s, [&] { return first_entered; }));
    }

    grpc::ClientContext queued_context;
    multimodal_inference::EmotionResponse queued_response;
    grpc::Status queued_status;
    std::thread queued_client([&] {
        queued_status = stub->PredictEmotion(&queued_context, Request(), &queued_response);
    });
    const auto queued_deadline = std::chrono::steady_clock::now() + 2s;
    while (runtime->Snapshot().inflight_calls != 2 &&
           std::chrono::steady_clock::now() < queued_deadline) {
        std::this_thread::sleep_for(2ms);
    }
    ASSERT_EQ(runtime->Snapshot().inflight_calls, 2u);

    std::thread shutdown_thread([&] {
        pool->Shutdown(false);
    });
    queued_client.join();
    EXPECT_EQ(queued_status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(handler_calls.load(std::memory_order_relaxed), 1);

    {
        std::lock_guard lock(mutex);
        release_first = true;
    }
    release_cv.notify_all();
    first_client.join();
    shutdown_thread.join();
    EXPECT_TRUE(first_status.ok()) << first_status.error_message();

    const auto snapshot = runtime->Snapshot();
    EXPECT_EQ(snapshot.inflight_calls, 0);
    EXPECT_EQ(snapshot.completed_calls, 2);
    EXPECT_EQ(snapshot.failed_calls, 1);
    pool->Shutdown(true);
}

TEST(AsyncGrpcRuntimeTest, FailsImmediatelyWhenThreadPoolIsUnavailable) {
    core::ThreadPoolOptions pool_options;
    pool_options.worker_count = 1;
    auto pool = std::make_shared<core::ThreadPool>(std::move(pool_options));
    auto runtime = CreateRuntime(pool);
    auto handler = std::make_shared<LambdaEmotionHandler>(
        [](const auto&, const auto&, auto&) { return core::Status::Ok(); });
    server::grpc_service::AsyncEmotionGrpcService service(runtime, handler);
    GrpcServerHarness server(service);
    auto stub = multimodal_inference::MultimodalInference::NewStub(server.channel());

    grpc::ClientContext context;
    multimodal_inference::EmotionResponse response;
    const auto status = stub->PredictEmotion(&context, Request(), &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
    const auto snapshot = runtime->Snapshot();
    EXPECT_EQ(snapshot.submission_failures, 1);
    EXPECT_EQ(snapshot.inflight_calls, 0);
}

} // namespace
