#include "shared_memory_media_runtime.h"

#include "inference_frame_coordinator.h"
#include "inference_frame_ipc_receiver.h"
#include "inference_frame_spool.h"
#include "inference_frame_spool_replayer.h"
#include "media_inference_execution.h"
#include "memory_pool.h"
#include "ordered_inference_frame_admission.h"
#include "skill_session_manager.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

namespace service {
namespace {

using Execution = agent::service::persona::MediaInferenceExecution;
using ExecutionSnapshot = agent::service::persona::MediaInferenceExecutionSnapshot;
using ExecutionState = agent::service::persona::MediaInferenceExecutionState;

constexpr std::string_view kDefaultVisionPrompt =
    "请只输出一个 JSON 对象，字段为 scene_hint、action_hint、object_hint、facts、"
    "weak_interpretations、memory_candidate、confidence。只描述画面中直接可见的场景、动作和物体；"
    "不要推断人物身份、性别、情绪、关系或不可见事实。facts 和 weak_interpretations 是字符串数组，"
    "confidence 是 0 到 1 的数字。无法确认的内容放入 weak_interpretations。不要输出 Markdown。";

const char* StateName(ExecutionState state) noexcept {
    switch (state) {
    case ExecutionState::Running: return "running";
    case ExecutionState::SealingInput: return "sealing";
    case ExecutionState::ReplayingSpool: return "replaying";
    case ExecutionState::Aggregating: return "aggregating";
    case ExecutionState::Closed: return "closed";
    case ExecutionState::Failed: return "failed";
    }
    return "unknown";
}

std::string StripJsonFence(std::string text) {
    const auto first = text.find('{');
    const auto last = text.rfind('}');
    if (first != std::string::npos && last != std::string::npos && last >= first) {
        return text.substr(first, last - first + 1);
    }
    return text;
}

std::string JsonString(const nlohmann::json& value, std::string_view key) {
    const auto it = value.find(std::string(key));
    return it != value.end() && it->is_string() ? it->get<std::string>() : std::string{};
}

std::vector<std::string> JsonStrings(const nlohmann::json& value, std::string_view key) {
    std::vector<std::string> result;
    const auto it = value.find(std::string(key));
    if (it == value.end() || !it->is_array()) return result;
    for (const auto& item : *it) {
        if (item.is_string()) result.push_back(item.get<std::string>());
    }
    return result;
}

media::VisionInferenceResult ParseResponse(const multimodal_inference::VLMResponse& response) {
    media::VisionInferenceResult result;
    result.raw_text = response.text();
    result.image_encode_ms = response.image_encode_ms();
    result.prompt_eval_ms = response.prompt_eval_ms();
    result.eval_ms = response.eval_ms();
    result.prompt_tokens = response.prompt_tokens();
    result.generated_tokens = response.generated_tokens();
    result.cache_hit = response.cache_hit();
    result.cache_stale = response.cache_stale();
    result.prompt_kv_cache_hit = response.prompt_kv_cache_hit();
    result.result_source = response.result_source();
    result.prompt_kv_near_candidate = response.prompt_kv_near_candidate();
    result.prompt_kv_near_accepted = response.prompt_kv_near_accepted();
    result.prompt_kv_near_same_session = response.prompt_kv_near_same_session();
    result.prompt_kv_global_cosine = response.prompt_kv_global_cosine();
    result.prompt_kv_mean_token_cosine = response.prompt_kv_mean_token_cosine();
    result.prompt_kv_p05_token_cosine = response.prompt_kv_p05_token_cosine();
    result.prompt_kv_min_token_cosine = response.prompt_kv_min_token_cosine();
    result.prompt_kv_relative_l2 = response.prompt_kv_relative_l2();
    result.prompt_kv_max_abs_error = response.prompt_kv_max_abs_error();

    const auto parsed = nlohmann::json::parse(StripJsonFence(response.text()), nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        result.scene_hint = response.text();
        result.weak_interpretations.push_back("VLM 返回了非结构化观察文本");
        result.confidence = response.text().empty() ? 0.0 : 0.5;
        return result;
    }
    result.scene_hint = JsonString(parsed, "scene_hint");
    result.action_hint = JsonString(parsed, "action_hint");
    result.object_hint = JsonString(parsed, "object_hint");
    result.agent_hint = JsonString(parsed, "agent_hint");
    result.memory_candidate = JsonString(parsed, "memory_candidate");
    result.facts = JsonStrings(parsed, "facts");
    result.weak_interpretations = JsonStrings(parsed, "weak_interpretations");
    if (const auto it = parsed.find("confidence"); it != parsed.end() && it->is_number()) {
        result.confidence = std::clamp(it->get<double>(), 0.0, 1.0);
    }
    if (result.scene_hint.empty() && !result.facts.empty()) result.scene_hint = result.facts.front();
    return result;
}

class LocalVlmClient final : public media::IVlmVisionClient {
public:
    LocalVlmClient(IMultimodalService& service, const SharedMemoryMediaRuntimeOptions& options)
        : service_(service), options_(options) {}

    core::Result<media::VisionInferenceResult> Analyze(
        const media::VisionInferenceRequest& input) override {
        if (input.session_id.empty() || input.encoded_image.empty()) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "shared media VLM request identity or image is missing");
        }
        multimodal_inference::VLMRequest request;
        // protobuf service 目前接收 owned bytes；共享 slot 已在 receiver 中及时 ack。
        request.set_image_data(input.encoded_image.data(), input.encoded_image.size());
        request.set_prompt(input.prompt_hint.value_or(std::string(kDefaultVisionPrompt)));
        request.set_max_tokens(options_.max_tokens);
        request.set_context_size(options_.context_size);
        request.set_temperature(options_.temperature);
        request.set_top_p(options_.top_p);
        request.set_top_k(options_.top_k);
        request.set_session_id(input.session_id);
        request.set_request_id(input.session_id + "-frame-" + std::to_string(input.frame_id));
        request.set_task_type("shared-memory-vision-observation");
        request.set_allow_cache(options_.allow_cache);
        request.set_force_refresh(!options_.allow_cache);

        multimodal_inference::VLMResponse response;
        auto status = service_.GenerateVLMSync(request, response);
        if (!status.ok()) return status;
        if (!response.error().empty()) {
            return core::Status::Error(core::ErrorCode::InternalError, response.error());
        }
        return ParseResponse(response);
    }

private:
    IMultimodalService& service_;
    const SharedMemoryMediaRuntimeOptions& options_;
};

class ExecutionRouter final : public media::inference::IInferenceFrameAdmissionSink {
public:
    core::Status Register(
        std::string execution_id,
        std::string session_id,
        std::shared_ptr<Execution> execution) {
        if (execution_id.empty() || session_id.empty() || !execution) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "media execution route is invalid");
        }
        std::lock_guard lock(mutex_);
        if (routes_.contains(execution_id)) {
            return core::Status::Error(core::ErrorCode::AlreadyExists, "media execution route already exists");
        }
        routes_.emplace(std::move(execution_id), Route{std::move(session_id), execution});
        return core::Status::Ok();
    }

    core::Status AdmitFrame(media::inference::OwnedInferenceFrame frame) override {
        const auto metadata = frame.metadata();
        std::shared_ptr<Execution> execution;
        {
            std::lock_guard lock(mutex_);
            const auto it = routes_.find(metadata.execution_id);
            if (it == routes_.end()) {
                return core::Status::Error(core::ErrorCode::NotFound, "media execution route not found");
            }
            if (it->second.session_id != metadata.session_id) {
                return core::Status::Error(
                    core::ErrorCode::FailedPrecondition,
                    "media execution route belongs to another session");
            }
            execution = it->second.execution.lock();
        }
        if (!execution) {
            return core::Status::Error(core::ErrorCode::Cancelled, "media execution route expired");
        }
        return execution->AdmitFrame(std::move(frame));
    }

    std::shared_ptr<Execution> Find(std::string_view execution_id) const {
        std::lock_guard lock(mutex_);
        const auto it = routes_.find(std::string(execution_id));
        return it == routes_.end() ? nullptr : it->second.execution.lock();
    }

private:
    struct Route {
        std::string session_id;
        std::weak_ptr<Execution> execution;
    };
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Route> routes_;
};

} // namespace

class SharedMemoryMediaRuntime::Impl {
public:
    struct ExecutionContext {
        std::string execution_id;
        std::string session_id;
        std::string skill_id;
        std::shared_ptr<media::inference::IInferenceFrameSpool> spool;
        std::shared_ptr<media::inference::IInferenceFrameSpoolReplayer> replayer;
        std::shared_ptr<Execution> execution;
        mutable std::mutex mutex;
        std::size_t expected_selected_frames = 0;
        std::uint64_t final_transport_sequence = 0;
        bool seal_requested = false;
        bool seal_task_started = false;
        std::int64_t input_sealed_at_unix_us = 0;
        core::Status control_status = core::Status::Ok();
        std::optional<agent::service::persona::MediaInferenceExecutionCompletion> completion;
    };

    Impl(
        IMultimodalService& inference_service,
        ipc::media::IInferenceFrameIpcGrantReceiver& ipc_source,
        SharedMemoryMediaRuntimeOptions options,
        core::LoggerAdapter logger)
        : inference_service_(inference_service),
          ipc_source_(ipc_source),
          options_(std::move(options)),
          logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("shared-media-runtime")) {}

    ~Impl() { Shutdown(); }

    core::Status Start() {
        if (options_.receiver_workers == 0 || options_.vlm_workers == 0 || options_.max_executions == 0 ||
            options_.backlog_segments_per_session == 0 || options_.backlog_slots_per_segment == 0 ||
            options_.spool_segment_bytes == 0 || options_.max_spool_bytes_per_execution == 0) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "shared media runtime limits are invalid");
        }
        std::error_code ec;
        std::filesystem::create_directories(options_.spool_root, ec);
        if (ec) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to create shared media spool root");
        }

        backlog_ = std::make_shared<media::inference::SegmentedInferenceFrameBacklog>(
            media::inference::SegmentedFrameBacklogOptions{
                .max_sessions = options_.max_executions,
                .segments_per_session = options_.backlog_segments_per_session,
                .slots_per_segment = options_.backlog_slots_per_segment,
                .default_wait_timeout = std::chrono::milliseconds(20),
            });
        result_table_ = std::make_shared<media::inference::SessionInferenceFrameResultTable>(
            media::inference::InferenceFrameResultTableOptions{
                .max_sessions = options_.max_executions,
                .max_results_per_session = options_.max_results_per_execution,
            });
        auto runtime = agent::service::persona::MediaInferenceExecutionRuntime::Create({
            .io_pool = {2, options_.max_executions * 8, "shared-media-spool-io"},
            .control_pool = {2, options_.max_executions * 2, "shared-media-control"},
            .aggregation_pool = {1, options_.max_executions, "shared-media-aggregation"},
        });
        if (!runtime.ok()) return runtime.status();
        execution_runtime_ = std::move(runtime).value();
        skill_sessions_ = std::make_shared<agent::service::persona::SkillSessionManager>(
            agent::service::persona::SkillSessionOptions{
                .startup_timeout = std::chrono::seconds(30),
                .max_duration = std::chrono::minutes(30),
                .idle_timeout = std::chrono::minutes(10),
                .closing_timeout = std::chrono::minutes(30),
                .max_recent_observations = 8,
            });
        router_ = std::make_shared<ExecutionRouter>();
        ordered_admission_ = std::make_shared<media::inference::OrderedInferenceFrameAdmission>(
            router_,
            media::inference::OrderedInferenceFrameAdmissionOptions{
                .max_executions = options_.max_executions,
                .window_capacity = options_.max_results_per_execution,
            });
        receiver_ = std::make_unique<media::inference::InferenceFrameIpcReceiver>(
            ipc_source_,
            memory_pool_,
            *backlog_,
            logger_,
            [this](const auto& metadata, const auto& status) {
                if (!status.ok() && status.code() != core::ErrorCode::NotFound &&
                    status.code() != core::ErrorCode::Cancelled) {
                    logger_.warn(
                        "[shared-media-runtime] receiver rejected execution={} session={} frame={} code={} message={}",
                        metadata.execution_id,
                        metadata.session_id,
                        metadata.frame_id,
                        static_cast<int>(status.code()),
                        status.message());
                }
            },
            media::inference::InferenceFrameIpcReceiverOptions{
                .admission_sink = ordered_admission_,
            });
        vlm_client_ = std::make_unique<LocalVlmClient>(inference_service_, options_);
        coordinator_ = std::make_unique<media::inference::InferenceFrameCoordinator>(
            *backlog_,
            *vlm_client_,
            *result_table_,
            media::inference::InferenceFrameCoordinatorOptions{
                .worker_count = options_.vlm_workers,
                .wait_timeout = std::chrono::milliseconds(20),
                .shutdown_backlog = false,
                .terminal_observer = [this](auto event) { ObserveTerminal(std::move(event)); },
            },
            logger_);
        auto status = coordinator_->Start();
        if (!status.ok()) return status;

        running_.store(true, std::memory_order_release);
        receiver_threads_.reserve(options_.receiver_workers);
        for (std::size_t index = 0; index < options_.receiver_workers; ++index) {
            receiver_threads_.emplace_back([this](std::stop_token stop_token) { ReceiverLoop(stop_token); });
        }
        return core::Status::Ok();
    }

    core::Result<SharedMediaExecutionSnapshot> Open(
        const SharedMediaExecutionOpenRequest& request) {
        if (request.execution_id.empty() || request.session_id.empty()) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "execution_id and session_id are required");
        }
        auto context = std::make_shared<ExecutionContext>();
        context->execution_id = request.execution_id;
        context->session_id = request.session_id;
        context->skill_id = request.skill_id.empty() ? "vision.observe" : request.skill_id;
        {
            std::lock_guard lock(executions_mutex_);
            if (executions_.contains(request.execution_id)) {
                return core::Status::Error(core::ErrorCode::AlreadyExists, "shared media execution already exists");
            }
            if (executions_.size() >= options_.max_executions) {
                return core::Status::Error(core::ErrorCode::ResourceExhausted, "shared media execution limit reached");
            }
        }

        auto spool = media::inference::MappedInferenceFrameSpool::Create({
            .root_directory = options_.spool_root,
            .execution_id = request.execution_id,
            .segment_bytes = options_.spool_segment_bytes,
            .max_spool_bytes = options_.max_spool_bytes_per_execution,
            .flush_on_append = false,
            .remove_on_destroy = true,
        }, logger_);
        if (!spool.ok()) return spool.status();
        context->spool = std::shared_ptr<media::inference::IInferenceFrameSpool>(std::move(spool).value());
        context->replayer = std::make_shared<media::inference::InferenceFrameSpoolReplayer>(
            *context->spool,
            memory_pool_,
            *backlog_,
            media::inference::InferenceFrameSpoolReplayOptions{16},
            logger_);

        agent::service::persona::SkillSessionStartRequest start;
        start.execution_id = request.execution_id;
        start.session_id = request.session_id;
        start.skill_id = context->skill_id;
        start.trace_id = request.trace_id;
        start.source = "shared-memory-ipc";
        start.reason = "shared media inference opened";
        auto started = skill_sessions_->Start(start);
        if (!started.ok()) return started.status();
        auto status = skill_sessions_->MarkReady(
            request.session_id,
            context->skill_id,
            "shared media inference ready",
            request.trace_id);
        if (!status.ok()) return status;

        auto execution = Execution::Create(
            {
                .execution_id = request.execution_id,
                .session_id = request.session_id,
                .skill_id = context->skill_id,
                .trace_id = request.trace_id,
                .replay_retry_delay = std::chrono::milliseconds(1),
            },
            {
                .runtime = execution_runtime_,
                .backlog = backlog_,
                .spool = context->spool,
                .replayer = context->replayer,
                .result_table = result_table_,
                .skill_sessions = skill_sessions_,
            },
            [this, weak_context = std::weak_ptr<ExecutionContext>(context)](const auto& completion) {
                auto locked = weak_context.lock();
                if (!locked) {
                    return core::Status::Error(core::ErrorCode::Cancelled, "shared media execution context expired");
                }
                std::lock_guard lock(locked->mutex);
                locked->completion = completion;
                return core::Status::Ok();
            },
            logger_);
        if (!execution.ok()) return execution.status();
        context->execution = std::move(execution).value();
        status = router_->Register(request.execution_id, request.session_id, context->execution);
        if (!status.ok()) return status;
        {
            std::lock_guard lock(executions_mutex_);
            executions_.emplace(request.execution_id, context);
        }
        return Snapshot(*context, false);
    }

    core::Result<SharedMediaExecutionSnapshot> Seal(
        const SharedMediaExecutionSealRequest& request) {
        if (request.execution_id.empty() || request.session_id.empty() ||
            request.expected_selected_frames != request.final_transport_sequence) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "shared media seal identity or expected frame fence is invalid");
        }
        auto context = Find(request.session_id, request.execution_id);
        if (!context.ok()) return context.status();
        bool already_sealed = false;
        {
            std::lock_guard lock(context.value()->mutex);
            if (context.value()->seal_requested) {
                if (context.value()->expected_selected_frames != request.expected_selected_frames ||
                    context.value()->final_transport_sequence != request.final_transport_sequence) {
                    return core::Status::Error(
                        core::ErrorCode::FailedPrecondition,
                        "shared media execution was sealed with another frame fence");
                }
                already_sealed = true;
            } else {
                context.value()->seal_requested = true;
                context.value()->input_sealed_at_unix_us = media::inference::InferenceFrameNowUnixUs();
                context.value()->expected_selected_frames = request.expected_selected_frames;
                context.value()->final_transport_sequence = request.final_transport_sequence;
                context.value()->seal_task_started = true;
            }
        }
        if (already_sealed) return Snapshot(*context.value(), false);
        auto status = execution_runtime_->control_pool().Submit(
            [this, context = context.value(), reason = request.reason] {
                return CompleteSeal(std::move(context), reason);
            },
            {},
            "shared-media-seal");
        if (!status.ok()) {
            std::lock_guard lock(context.value()->mutex);
            context.value()->control_status = status;
            return status;
        }
        return Snapshot(*context.value(), false);
    }

    core::Result<SharedMediaExecutionSnapshot> Get(
        std::string_view session_id,
        std::string_view execution_id,
        bool include_results) const {
        auto context = Find(session_id, execution_id);
        if (!context.ok()) return context.status();
        return Snapshot(*context.value(), include_results);
    }

    void Shutdown() {
        if (!running_.exchange(false, std::memory_order_acq_rel)) return;
        for (auto& thread : receiver_threads_) thread.request_stop();
        receiver_threads_.clear();
        if (receiver_) receiver_->Shutdown();
        if (coordinator_) coordinator_->Shutdown();
        if (execution_runtime_) execution_runtime_->Shutdown(true);
        if (result_table_) result_table_->Shutdown();
        if (backlog_) backlog_->Shutdown();
    }

private:
    void ReceiverLoop(std::stop_token stop_token) {
        while (!stop_token.stop_requested() && running_.load(std::memory_order_acquire)) {
            const auto status = receiver_->PollOnce();
            if (status.ok()) continue;
            if (status.code() != core::ErrorCode::NotFound &&
                status.code() != core::ErrorCode::Cancelled) {
                logger_.warn(
                    "[shared-media-runtime] receiver poll failed code={} message={}",
                    static_cast<int>(status.code()),
                    status.message());
            }
            std::this_thread::sleep_for(options_.receiver_idle_delay);
        }
    }

    void ObserveTerminal(media::inference::InferenceFrameTerminalEvent event) {
        auto execution = router_->Find(event.frame.execution_id);
        if (!execution) {
            logger_.warn(
                "[shared-media-runtime] terminal route missing execution={} frame={}",
                event.frame.execution_id,
                event.frame.frame_id);
            return;
        }
        const auto status = execution->ObserveTerminal(std::move(event));
        if (!status.ok()) {
            logger_.warn(
                "[shared-media-runtime] terminal observation failed code={} message={}",
                static_cast<int>(status.code()),
                status.message());
        }
    }

    core::Status CompleteSeal(std::shared_ptr<ExecutionContext> context, std::string reason) {
        const auto deadline = std::chrono::steady_clock::now() + options_.seal_wait_timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            const auto snapshot = context->execution->Snapshot();
            std::size_t expected = 0;
            std::uint64_t final_transport = 0;
            {
                std::lock_guard lock(context->mutex);
                expected = context->expected_selected_frames;
                final_transport = context->final_transport_sequence;
            }
            if (snapshot.selected_frames >= expected) {
                auto status = ordered_admission_->SealExecution(
                    context->session_id,
                    context->execution_id,
                    final_transport);
                if (!status.ok()) {
                    std::lock_guard lock(context->mutex);
                    context->control_status = status;
                    return status;
                }
                status = context->execution->BeginClosing(
                    reason.empty() ? "shared media input sealed" : std::move(reason));
                if (!status.ok()) {
                    std::lock_guard lock(context->mutex);
                    context->control_status = status;
                }
                return status;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto status = core::Status::Error(
            core::ErrorCode::Timeout,
            "shared media seal timed out waiting for admitted frames");
        {
            std::lock_guard lock(context->mutex);
            context->control_status = status;
        }
        skill_sessions_->MarkFailed(
            context->session_id,
            context->skill_id,
            status.message(),
            {},
            context->execution_id);
        return status;
    }

    core::Result<std::shared_ptr<ExecutionContext>> Find(
        std::string_view session_id,
        std::string_view execution_id) const {
        std::lock_guard lock(executions_mutex_);
        const auto it = executions_.find(std::string(execution_id));
        if (it == executions_.end()) {
            return core::Status::Error(core::ErrorCode::NotFound, "shared media execution not found");
        }
        if (it->second->session_id != session_id) {
            return core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "shared media execution belongs to another session");
        }
        return it->second;
    }

    SharedMediaExecutionSnapshot Snapshot(
        const ExecutionContext& context,
        bool include_results) const {
        const ExecutionSnapshot execution = context.execution->Snapshot();
        SharedMediaExecutionSnapshot snapshot;
        snapshot.execution_id = context.execution_id;
        snapshot.session_id = context.session_id;
        snapshot.state = StateName(execution.state);
        snapshot.selected_frames = execution.selected_frames;
        snapshot.committed_frames = execution.committed_frames;
        snapshot.hot_frames = execution.hot_frames;
        snapshot.spooled_frames = execution.spooled_frames;
        snapshot.terminal_frames = execution.terminal_frames;
        snapshot.failed_inference_frames = execution.failed_inference_frames;
            snapshot.replay_complete = execution.replay_complete;
            snapshot.complete = execution.state == ExecutionState::Closed || execution.state == ExecutionState::Failed;
            snapshot.status = execution.terminal_status;
        {
            std::lock_guard lock(context.mutex);
            snapshot.expected_selected_frames = context.expected_selected_frames;
            snapshot.seal_requested = context.seal_requested;
            snapshot.input_sealed_at_unix_us = context.input_sealed_at_unix_us;
            if (!context.control_status.ok()) snapshot.status = context.control_status;
            if (include_results && context.completion.has_value()) {
                snapshot.results = context.completion->results;
                if (!context.completion->status.ok()) snapshot.status = context.completion->status;
            }
        }
        return snapshot;
    }

    IMultimodalService& inference_service_;
    ipc::media::IInferenceFrameIpcGrantReceiver& ipc_source_;
    SharedMemoryMediaRuntimeOptions options_;
    core::LoggerAdapter logger_;
    core::BucketMemoryPool memory_pool_;
    std::shared_ptr<media::inference::SegmentedInferenceFrameBacklog> backlog_;
    std::shared_ptr<media::inference::SessionInferenceFrameResultTable> result_table_;
    std::shared_ptr<agent::service::persona::MediaInferenceExecutionRuntime> execution_runtime_;
    std::shared_ptr<agent::service::persona::SkillSessionManager> skill_sessions_;
    std::shared_ptr<ExecutionRouter> router_;
    std::shared_ptr<media::inference::OrderedInferenceFrameAdmission> ordered_admission_;
    std::unique_ptr<media::inference::InferenceFrameIpcReceiver> receiver_;
    std::unique_ptr<LocalVlmClient> vlm_client_;
    std::unique_ptr<media::inference::InferenceFrameCoordinator> coordinator_;
    std::atomic<bool> running_{false};
    std::vector<std::jthread> receiver_threads_;
    mutable std::mutex executions_mutex_;
    std::unordered_map<std::string, std::shared_ptr<ExecutionContext>> executions_;
};

core::Result<std::unique_ptr<SharedMemoryMediaRuntime>> SharedMemoryMediaRuntime::Create(
    IMultimodalService& inference_service,
    ipc::media::IInferenceFrameIpcGrantReceiver& ipc_source,
    SharedMemoryMediaRuntimeOptions options,
    core::LoggerAdapter logger) {
    auto impl = std::make_unique<Impl>(
        inference_service,
        ipc_source,
        std::move(options),
        std::move(logger));
    auto status = impl->Start();
    if (!status.ok()) {
        impl->Shutdown();
        return status;
    }
    return std::unique_ptr<SharedMemoryMediaRuntime>(
        new SharedMemoryMediaRuntime(std::move(impl)));
}

SharedMemoryMediaRuntime::SharedMemoryMediaRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

SharedMemoryMediaRuntime::~SharedMemoryMediaRuntime() = default;

core::Result<SharedMediaExecutionSnapshot> SharedMemoryMediaRuntime::Open(
    const SharedMediaExecutionOpenRequest& request) {
    return impl_->Open(request);
}

core::Result<SharedMediaExecutionSnapshot> SharedMemoryMediaRuntime::Seal(
    const SharedMediaExecutionSealRequest& request) {
    return impl_->Seal(request);
}

core::Result<SharedMediaExecutionSnapshot> SharedMemoryMediaRuntime::Get(
    std::string_view session_id,
    std::string_view execution_id,
    bool include_results) const {
    return impl_->Get(session_id, execution_id, include_results);
}

void SharedMemoryMediaRuntime::Shutdown() {
    impl_->Shutdown();
}

} // namespace service
