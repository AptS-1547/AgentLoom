#include "../../src/skill/skill_registry.h"
#include "../../src/skill/skill_executor.h"

#include <gtest/gtest.h>

#include <future>

namespace {

agent::skill::SkillManifest MakeManifest() {
    agent::skill::SkillManifest manifest;
    manifest.skill_id = "vision.observe";
    manifest.version = "1.0.0";
    manifest.input_schema_json = R"({"type":"object","required":["reason"]})";
    manifest.executor.reference = manifest.skill_id;
    return manifest;
}

TEST(SkillManifestTest, RejectsInvalidSchema) {
    auto manifest = MakeManifest();
    manifest.input_schema_json = "{";
    EXPECT_EQ(agent::skill::ValidateManifest(manifest).code(),
              core::ErrorCode::InvalidArgument);
}

TEST(SkillRegistryTest, RegistersAndFindsVersionedManifest) {
    agent::skill::InMemorySkillRegistry registry;
    ASSERT_TRUE(registry.Register(MakeManifest()).ok());
    auto found = registry.Require("vision.observe", "1.0.0");
    ASSERT_TRUE(found.ok()) << found.status().message();
    EXPECT_EQ(found.value().skill_id, "vision.observe");
}

TEST(SkillRegistryTest, RejectsDuplicateManifest) {
    agent::skill::InMemorySkillRegistry registry;
    ASSERT_TRUE(registry.Register(MakeManifest()).ok());
    EXPECT_EQ(registry.Register(MakeManifest()).code(), core::ErrorCode::AlreadyExists);
}

TEST(SkillRegistryTest, RegistersJsonManifest) {
    agent::skill::InMemorySkillRegistry registry;
    ASSERT_TRUE(registry.RegisterJson(R"({"skill_id":"crm.lookup","version":"1.0.0","executor":{"type":"native","reference":"crm.lookup"}})").ok());
    EXPECT_TRUE(registry.Require("crm.lookup", "1.0.0").ok());
}

TEST(SkillRegistryTest, RegistersAllManifestsAtomicallyForStartupValidation) {
    agent::skill::InMemorySkillRegistry registry;
    auto first = MakeManifest();
    auto second = MakeManifest();
    second.skill_id = "document.analyze";
    second.executor.reference = second.skill_id;
    ASSERT_TRUE(registry.RegisterAll({first, second}).ok());
    EXPECT_TRUE(registry.Require("vision.observe").ok());
    EXPECT_TRUE(registry.Require("document.analyze").ok());
}

TEST(SkillExecutorFactoryTest, ResolvesRegisteredExecutorType) {
    class FakeExecutor final : public agent::skill::ISkillExecutor {
    public:
        core::Result<agent::service::persona::SkillSessionSnapshot> Start(
            const agent::skill::SkillExecutionRequest&, agent::skill::SkillExecutionCallbacks) override {
            return core::Status::Error(core::ErrorCode::Unavailable, "fake");
        }
        core::Status Cancel(std::string_view) override { return core::Status::Ok(); }
    };
    agent::skill::InMemorySkillExecutorFactory factory;
    auto executor = std::make_shared<FakeExecutor>();
    ASSERT_TRUE(factory.Register("native", executor).ok());
    auto manifest = MakeManifest();
    auto resolved = factory.Resolve(manifest);
    ASSERT_TRUE(resolved.ok()) << resolved.status().message();
    EXPECT_EQ(resolved.value(), executor);
}

TEST(SkillInvocationServiceTest, RoutesObservationAndResultThroughSessionLifecycle) {
    class CompletingExecutor final : public agent::skill::ISkillExecutor {
    public:
        core::Result<agent::service::persona::SkillSessionSnapshot> Start(
            const agent::skill::SkillExecutionRequest& request,
            agent::skill::SkillExecutionCallbacks callbacks) override {
            if (callbacks.on_observation) {
                agent::service::persona::SkillObservation observation;
                observation.session_id = request.session_id;
                observation.skill_id = request.call.skill_id;
                observation.summary = "completed";
                callbacks.on_observation(std::move(observation));
            }
            if (callbacks.on_result) {
                agent::skill::SkillResult result;
                result.call_id = request.call.call_id;
                result.result_json = "{\"ok\":true}";
                callbacks.on_result(std::move(result));
            }
            return agent::service::persona::SkillSessionSnapshot{};
        }
        core::Status Cancel(std::string_view) override { return core::Status::Ok(); }
    };
    auto registry = std::make_shared<agent::skill::InMemorySkillRegistry>();
    auto manifest = MakeManifest();
    ASSERT_TRUE(registry->Register(manifest).ok());
    auto factory = std::make_shared<agent::skill::InMemorySkillExecutorFactory>();
    ASSERT_TRUE(factory->RegisterReference("native", manifest.skill_id,
                                           std::make_shared<CompletingExecutor>()).ok());
    auto sessions = std::make_shared<agent::service::persona::SkillSessionManager>();
    agent::skill::SkillInvocationService service(registry, factory, sessions);
    agent::skill::SkillExecutionRequest request;
    request.call = {"call-1", {}, manifest.skill_id, manifest.version, "{}"};
    request.session_id = "session-1";
    auto invoked = service.Invoke(std::move(request), {});
    ASSERT_TRUE(invoked.ok()) << invoked.status().message();
    auto snapshot = sessions->Get("session-1", manifest.skill_id);
    ASSERT_TRUE(snapshot.ok());
    ASSERT_TRUE(snapshot.value().has_value());
    EXPECT_EQ(snapshot.value()->state, agent::service::persona::SkillSessionState::Closed);
}

TEST(SkillToolCallCoordinatorTest, ExecutesStandardToolCallAndBuildsToolResult) {
    class ImmediateExecutor final : public agent::skill::ISkillExecutor {
    public:
        core::Result<agent::service::persona::SkillSessionSnapshot> Start(
            const agent::skill::SkillExecutionRequest& request,
            agent::skill::SkillExecutionCallbacks callbacks) override {
            agent::skill::SkillResult result;
            result.call_id = request.call.call_id;
            result.result_json = R"({"value":42})";
            callbacks.on_result(std::move(result));
            return agent::service::persona::SkillSessionSnapshot{};
        }
        core::Status Cancel(std::string_view) override { return core::Status::Ok(); }
    };
    auto registry = std::make_shared<agent::skill::InMemorySkillRegistry>();
    auto manifest = MakeManifest();
    ASSERT_TRUE(registry->Register(manifest).ok());
    auto factory = std::make_shared<agent::skill::InMemorySkillExecutorFactory>();
    ASSERT_TRUE(factory->RegisterReference("native", manifest.skill_id,
                                           std::make_shared<ImmediateExecutor>()).ok());
    auto sessions = std::make_shared<agent::service::persona::SkillSessionManager>();
    auto invocation = std::make_shared<agent::skill::SkillInvocationService>(registry, factory, sessions);
    agent::skill::SkillToolCallCoordinator coordinator(registry, invocation);
    agent::llm::ChatCompletionResponse response;
    response.tool_calls.push_back({"call-1", manifest.skill_id, R"({"reason":"look"})"});
    auto result = coordinator.Execute(response, {"session-1", "user-1", "persona-1", "trace-1", std::chrono::seconds(1)});
    ASSERT_TRUE(result.ok()) << result.status().message();
    ASSERT_EQ(result.value().size(), 1u);
    EXPECT_EQ(result.value()[0].role, agent::llm::ChatRole::Tool);
    EXPECT_EQ(result.value()[0].tool_call_id, "call-1");
    EXPECT_EQ(result.value()[0].content, R"({"value":42})");
}

TEST(SkillToolCallCoordinatorTest, CompletesTwoRoundToolCallingProtocol) {
    class ImmediateExecutor final : public agent::skill::ISkillExecutor {
    public:
        core::Result<agent::service::persona::SkillSessionSnapshot> Start(
            const agent::skill::SkillExecutionRequest& request,
            agent::skill::SkillExecutionCallbacks callbacks) override {
            agent::skill::SkillResult result;
            result.call_id = request.call.call_id;
            result.result_json = R"({"answer":"observed"})";
            callbacks.on_result(std::move(result));
            return agent::service::persona::SkillSessionSnapshot{};
        }
        core::Status Cancel(std::string_view) override { return core::Status::Ok(); }
    };
    auto registry = std::make_shared<agent::skill::InMemorySkillRegistry>();
    auto manifest = MakeManifest();
    ASSERT_TRUE(registry->Register(manifest).ok());
    auto factory = std::make_shared<agent::skill::InMemorySkillExecutorFactory>();
    ASSERT_TRUE(factory->RegisterReference("native", manifest.skill_id,
                                           std::make_shared<ImmediateExecutor>()).ok());
    auto sessions = std::make_shared<agent::service::persona::SkillSessionManager>();
    auto invocation = std::make_shared<agent::skill::SkillInvocationService>(registry, factory, sessions);
    agent::skill::SkillToolCallCoordinator coordinator(registry, invocation);

    agent::llm::ChatCompletionRequest request;
    request.tools.push_back({manifest.skill_id, "observe", manifest.input_schema_json});
    request.tool_choice = "auto";
    request.messages.push_back({agent::llm::ChatRole::User, "look"});
    agent::llm::ChatCompletionResponse first;
    first.tool_calls.push_back({"call-1", manifest.skill_id, R"({"reason":"look"})"});
    auto tool_messages = coordinator.Execute(first, {"session-2", "user-1", "persona-1", "trace-2", std::chrono::seconds(1)});
    ASSERT_TRUE(tool_messages.ok()) << tool_messages.status().message();
    ASSERT_EQ(tool_messages.value().size(), 1u);

    agent::llm::ChatMessage assistant;
    assistant.role = agent::llm::ChatRole::Assistant;
    assistant.tool_calls = first.tool_calls;
    request.messages.push_back(std::move(assistant));
    request.messages.push_back(tool_messages.value().front());
    request.messages.push_back({agent::llm::ChatRole::Assistant, "observed"});
    EXPECT_TRUE(agent::llm::ValidateChatCompletionRequest(request).ok());
    EXPECT_EQ(request.messages.back().content, "observed");
}

TEST(SkillToolCallCoordinatorTest, ConvertsExecutionTimeoutToToolErrorAndCancels) {
    class BlockingExecutor final : public agent::skill::ISkillExecutor {
    public:
        core::Result<agent::service::persona::SkillSessionSnapshot> Start(
            const agent::skill::SkillExecutionRequest&,
            agent::skill::SkillExecutionCallbacks) override {
            return agent::service::persona::SkillSessionSnapshot{};
        }
        core::Status Cancel(std::string_view) override {
            ++cancel_count;
            return core::Status::Ok();
        }
        int cancel_count = 0;
    };
    auto registry = std::make_shared<agent::skill::InMemorySkillRegistry>();
    auto manifest = MakeManifest();
    ASSERT_TRUE(registry->Register(manifest).ok());
    auto factory = std::make_shared<agent::skill::InMemorySkillExecutorFactory>();
    auto executor = std::make_shared<BlockingExecutor>();
    ASSERT_TRUE(factory->RegisterReference("native", manifest.skill_id, executor).ok());
    auto sessions = std::make_shared<agent::service::persona::SkillSessionManager>();
    auto invocation = std::make_shared<agent::skill::SkillInvocationService>(registry, factory, sessions);
    agent::skill::SkillToolCallCoordinator coordinator(registry, invocation);
    agent::llm::ChatCompletionResponse response;
    response.tool_calls.push_back({"timeout-call", manifest.skill_id, R"({"reason":"wait"})"});
    auto result = coordinator.Execute(response, {"session-timeout", "user", "persona", "trace", std::chrono::milliseconds(1)});
    ASSERT_TRUE(result.ok()) << result.status().message();
    ASSERT_EQ(result.value().size(), 1u);
    EXPECT_EQ(result.value()[0].role, agent::llm::ChatRole::Tool);
    EXPECT_NE(result.value()[0].content.find("timed out"), std::string::npos);
    EXPECT_EQ(executor->cancel_count, 1);
}

TEST(SkillToolCallCoordinatorTest, ExecutesAsyncWithoutBlockingUntilExecutorCompletes) {
    class DeferredExecutor final : public agent::skill::ISkillExecutor {
    public:
        core::Result<agent::service::persona::SkillSessionSnapshot> Start(
            const agent::skill::SkillExecutionRequest& request,
            agent::skill::SkillExecutionCallbacks callbacks) override {
            request_ = request;
            callbacks_ = std::move(callbacks);
            return agent::service::persona::SkillSessionSnapshot{};
        }
        core::Status Cancel(std::string_view) override { return core::Status::Ok(); }
        void Complete() {
            agent::skill::SkillResult result;
            result.call_id = request_.call.call_id;
            result.result_json = R"({"async":true})";
            callbacks_.on_result(std::move(result));
        }

    private:
        agent::skill::SkillExecutionRequest request_;
        agent::skill::SkillExecutionCallbacks callbacks_;
    };
    auto registry = std::make_shared<agent::skill::InMemorySkillRegistry>();
    auto manifest = MakeManifest();
    ASSERT_TRUE(registry->Register(manifest).ok());
    auto factory = std::make_shared<agent::skill::InMemorySkillExecutorFactory>();
    auto executor = std::make_shared<DeferredExecutor>();
    ASSERT_TRUE(factory->RegisterReference("native", manifest.skill_id, executor).ok());
    auto sessions = std::make_shared<agent::service::persona::SkillSessionManager>();
    auto invocation = std::make_shared<agent::skill::SkillInvocationService>(registry, factory, sessions);
    agent::skill::SkillToolCallCoordinator coordinator(registry, invocation);

    std::promise<core::Result<std::vector<agent::llm::ChatMessage>>> promise;
    auto future = promise.get_future();
    agent::llm::ChatCompletionResponse response;
    response.tool_calls.push_back({"async-call", manifest.skill_id, R"({"reason":"look"})"});
    ASSERT_TRUE(coordinator.ExecuteAsync(
        response,
        {"async-session", "user", "persona", "trace", std::chrono::seconds(1)},
        [&promise](auto result) mutable { promise.set_value(std::move(result)); }).ok());
    EXPECT_EQ(future.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    executor->Complete();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = future.get();
    ASSERT_TRUE(result.ok()) << result.status().message();
    ASSERT_EQ(result.value().size(), 1u);
    EXPECT_EQ(result.value()[0].role, agent::llm::ChatRole::Tool);
    EXPECT_EQ(result.value()[0].content, R"({"async":true})");
}

} // namespace
