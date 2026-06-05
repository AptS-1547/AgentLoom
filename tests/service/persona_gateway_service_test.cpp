#include "persona_gateway_http_adapter.h"
#include "persona_gateway_server.h"
#include "persona_gateway_service.h"
#include "http_server.h"
#include "redis_connection_pool.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <boost/asio.hpp>
#include <boost/beast.hpp>

#include <future>
#include <filesystem>
#include <fstream>
#include <mutex>

namespace {

using agent::service::gateway::ChatGatewayRequest;
using agent::service::gateway::AuthIdentity;
using agent::service::gateway::AuthRegistrationRequest;
using agent::service::gateway::AuthRegistrationResult;
using agent::service::gateway::ClassroomMessageGatewayRequest;
using agent::service::gateway::ClassroomProactiveGatewayRequest;
using agent::service::gateway::ClassroomPollGatewayRequest;
using agent::service::gateway::ClassroomScheduler;
using agent::service::gateway::CloseSessionGatewayRequest;
using agent::service::gateway::CreateSessionGatewayRequest;
using agent::service::gateway::PersonaGatewayHttpAdapter;
using agent::service::gateway::PersonaGatewayServer;
using agent::service::gateway::PersonaGatewayServerDependencies;
using agent::service::gateway::PersonaGatewayServerOptions;
using agent::service::gateway::PersonaGatewayService;
using agent::service::gateway::IGatewayAuthenticator;
using agent::service::gateway::IAuthRegistrationService;
using agent::service::gateway::SqliteAuthSessionStore;
using agent::service::gateway::RedisAuthSessionStore;
using agent::service::gateway::AuthSessionRecord;
using agent::service::gateway::TrainingReportGatewayRequest;
using agent::service::persona::NeutralEmotionAnalyzer;
using agent::service::persona::PersonaRuntime;
using agent::service::persona::PersonaRuntimeOptions;
using agent::service::persona::PersonalityConfig;
using agent::service::persona::SemanticMemoryContextProvider;
using agent::service::persona::SessionManager;
using Json = nlohmann::json;
namespace asio = boost::asio;
namespace beast = boost::beast;
using tcp = asio::ip::tcp;

class FakeSemanticCache final : public agent::semantic_cache::ISemanticCache {
public:
    core::Result<agent::semantic_cache::CacheLookupResult> Lookup(
        const agent::semantic_cache::CacheLookupRequest&) override {
        agent::semantic_cache::CacheLookupResult result;
        result.hit = true;
        result.payload = "remembered context";
        result.similarity_score = 0.95f;
        return result;
    }

    core::Status Store(const agent::semantic_cache::CacheStoreRequest& req) override {
        std::lock_guard lock(mutex_);
        stored_payloads.push_back(req.response_payload);
        return core::Status::Ok();
    }

    std::mutex mutex_;
    std::vector<std::string> stored_payloads;
};

class FakeLlmClient final : public agent::llm::ILlmClient {
public:
    core::Result<agent::llm::ChatCompletionResponse> Complete(
        const agent::llm::ChatCompletionRequest& req) override {
        std::lock_guard lock(mutex_);
        last_request = req;
        agent::llm::ChatCompletionResponse response;
        response.model = req.model;
        response.content = "student reply";
        response.total_tokens = 16;
        return response;
    }

    std::mutex mutex_;
    agent::llm::ChatCompletionRequest last_request;
};

class FixedAuthenticator final : public IGatewayAuthenticator {
public:
    explicit FixedAuthenticator(AuthIdentity identity)
        : identity_(std::move(identity)) {}

    core::Result<AuthIdentity> Authenticate(const ::net::BeastHttpRequest&) const override {
        return identity_;
    }

private:
    AuthIdentity identity_;
};

class RejectingAuthenticator final : public IGatewayAuthenticator {
public:
    core::Result<AuthIdentity> Authenticate(const ::net::BeastHttpRequest&) const override {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "auth token is missing");
    }
};

class FixedAuthRegistrationService final : public IAuthRegistrationService {
public:
    core::Result<AuthRegistrationResult> Register(const AuthRegistrationRequest& request) override {
        last_request = request;
        AuthRegistrationResult result;
        result.identity.user_uuid = request.user_uuid.empty() ? "generated-user-001" : request.user_uuid;
        result.identity.tenant_id = request.tenant_id.empty() ? "default" : request.tenant_id;
        result.identity.subject = request.subject.empty() ? result.identity.user_uuid : request.subject;
        result.identity.token_id = "token-001";
        result.identity.authenticated = true;
        result.issued_at = std::chrono::system_clock::now();
        result.identity.expires_at = result.issued_at + std::chrono::hours(1);
        result.token = "jwt-token";
        result.cookie_header = "agent_auth=jwt-token; Path=/; HttpOnly; SameSite=Lax";
        return result;
    }

    AuthRegistrationRequest last_request;
};

struct GatewayFixture {
    core::ThreadPool compute{{1, 64, "gateway-compute"}};
    core::ThreadPool io{{1, 64, "gateway-io"}};
    SessionManager sessions;
    std::shared_ptr<FakeSemanticCache> cache;
    std::shared_ptr<NeutralEmotionAnalyzer> emotion;
    std::shared_ptr<FakeLlmClient> llm;
    std::shared_ptr<SemanticMemoryContextProvider> memory;
    PersonaRuntime runtime;
    ClassroomScheduler classroom_scheduler;
    PersonaGatewayService gateway;

    GatewayFixture()
        : sessions(compute, io),
          cache(std::make_shared<FakeSemanticCache>()),
          emotion(std::make_shared<NeutralEmotionAnalyzer>()),
          llm(std::make_shared<FakeLlmClient>()),
          memory(std::make_shared<SemanticMemoryContextProvider>(cache)),
          runtime(sessions, memory, emotion, llm, PersonaRuntimeOptions{.recent_raw_turns = 10, .default_model = "test-model"}),
          gateway(sessions, runtime, &classroom_scheduler) {
        EXPECT_TRUE(compute.Start().ok());
        EXPECT_TRUE(io.Start().ok());
    }

    ~GatewayFixture() {
        compute.Shutdown(true);
        io.Shutdown(true);
    }
};

CreateSessionGatewayRequest MakeCreateRequest() {
    PersonalityConfig personality;
    personality.name = "dazhi";
    personality.description = "classroom student persona";

    CreateSessionGatewayRequest req;
    req.trace_id = "trace-create";
    req.session_id = "session-gateway";
    req.user_uuid = "user-gateway";
    req.persona_id = "dazhi";
    req.classroom_id = "classroom-a";
    req.context_ids = {"group_1"};
    req.default_persona = true;
    req.personality = std::move(personality);
    req.emotion_state_config.noise_sigma = 0.0;
    return req;
}

::net::BeastHttpResponse SendJsonRequest(std::uint16_t port,
                                         ::net::http::verb method,
                                         std::string target,
                                         Json body) {
    asio::io_context io;
    tcp::resolver resolver(io);
    beast::tcp_stream stream(io);
    stream.connect(resolver.resolve("127.0.0.1", std::to_string(port)));

    ::net::BeastHttpRequest req{method, std::move(target), 11};
    req.set(::net::http::field::host, "127.0.0.1");
    req.set(::net::http::field::content_type, "application/json");
    req.set("X-Trace-Id", body.value("traceId", "trace-http"));
    req.body() = body.dump();
    req.prepare_payload();
    ::net::http::write(stream, req);

    beast::flat_buffer buffer;
    ::net::BeastHttpResponse response;
    ::net::http::read(stream, buffer, response);
    beast::error_code ec;
    stream.socket().shutdown(tcp::socket::shutdown_both, ec);
    stream.socket().close(ec);
    return response;
}

TEST(PersonaGatewayServiceTest, RunsCreateChatReportAndCloseLifecycle) {
    GatewayFixture f;

    auto created = f.gateway.CreateSession(MakeCreateRequest());
    ASSERT_TRUE(created.ok()) << created.status().message();
    EXPECT_EQ(created.value().session.status, agent::service::persona::SessionStatus::Active);

    ChatGatewayRequest chat;
    chat.trace_id = "trace-chat";
    chat.session_id = "session-gateway";
    chat.persona_id = "dazhi";
    chat.message = "hello";
    auto reply = f.gateway.Chat(std::move(chat));
    ASSERT_TRUE(reply.ok()) << reply.status().message();
    EXPECT_EQ(reply.value().content, "student reply");
    EXPECT_EQ(reply.value().turn_index, 1u);
    EXPECT_TRUE(reply.value().l0_hit);

    TrainingReportGatewayRequest report;
    report.trace_id = "trace-report";
    report.session_id = "session-gateway";
    auto report_result = f.gateway.TrainingReport(std::move(report));
    ASSERT_TRUE(report_result.ok()) << report_result.status().message();
    EXPECT_EQ(report_result.value().total_turns, 1u);
    EXPECT_EQ(report_result.value().metrics.request_count, 1u);

    ClassroomProactiveGatewayRequest proactive;
    proactive.trace_id = "trace-proactive";
    proactive.classroom_id = "classroom-a";
    proactive.session_id = "session-gateway";
    proactive.persona_id = "dazhi";
    auto proactive_result = f.gateway.ClassroomProactive(std::move(proactive));
    ASSERT_TRUE(proactive_result.ok()) << proactive_result.status().message();
    EXPECT_TRUE(proactive_result.value().should_speak);
    EXPECT_EQ(proactive_result.value().turn_index, 2u);

    CloseSessionGatewayRequest close;
    close.trace_id = "trace-close";
    close.session_id = "session-gateway";
    auto closed = f.gateway.CloseSession(std::move(close));
    ASSERT_TRUE(closed.ok()) << closed.status().message();
    EXPECT_EQ(closed.value().session.status, agent::service::persona::SessionStatus::Closed);

    ChatGatewayRequest after_close;
    after_close.trace_id = "trace-after-close";
    after_close.session_id = "session-gateway";
    after_close.message = "again";
    auto rejected = f.gateway.Chat(std::move(after_close));
    EXPECT_FALSE(rejected.ok());
}

TEST(PersonaGatewayServiceTest, RoutesClassroomMessageByContextAndPollsProactiveState) {
    GatewayFixture f;

    auto created_a = f.gateway.CreateSession(MakeCreateRequest());
    ASSERT_TRUE(created_a.ok()) << created_a.status().message();

    auto req_b = MakeCreateRequest();
    req_b.trace_id = "trace-create-b";
    req_b.session_id = "session-b";
    req_b.persona_id = "xiaozhi";
    req_b.personality.name = "xiaozhi";
    req_b.context_ids = {"group_2"};
    req_b.default_persona = false;
    req_b.proactive_level = "medium";
    auto created_b = f.gateway.CreateSession(std::move(req_b));
    ASSERT_TRUE(created_b.ok()) << created_b.status().message();

    ClassroomMessageGatewayRequest message;
    message.trace_id = "trace-classroom-message";
    message.classroom_id = "classroom-a";
    message.context_id = "group_2";
    message.message = "hello xiaozhi";
    auto reply = f.gateway.ClassroomMessage(std::move(message));
    ASSERT_TRUE(reply.ok()) << reply.status().message();
    EXPECT_EQ(reply.value().session_id, "session-b");
    EXPECT_EQ(reply.value().speaker_persona_id, "xiaozhi");
    EXPECT_EQ(reply.value().classroom_id, "classroom-a");

    ClassroomPollGatewayRequest poll;
    poll.trace_id = "trace-poll";
    poll.classroom_id = "classroom-a";
    poll.persona_id = "xiaozhi";
    auto no_speak = f.gateway.ClassroomPoll(std::move(poll));
    ASSERT_TRUE(no_speak.ok()) << no_speak.status().message();
    EXPECT_FALSE(no_speak.value().should_speak);

    ClassroomPollGatewayRequest system_poll;
    system_poll.trace_id = "trace-system-poll";
    system_poll.classroom_id = "classroom-a";
    system_poll.persona_id = "xiaozhi";
    system_poll.system_event = true;
    system_poll.system_event_content = "teacher asks xiaozhi";
    auto proactive = f.gateway.ClassroomPoll(std::move(system_poll));
    ASSERT_TRUE(proactive.ok()) << proactive.status().message();
    EXPECT_TRUE(proactive.value().should_speak);
    EXPECT_EQ(proactive.value().speaker_persona_id, "xiaozhi");
}

TEST(PersonaGatewayHttpAdapterTest, HandlesSessionCreateAndChatJsonRoutes) {
    GatewayFixture f;
    PersonaGatewayHttpAdapter adapter(f.gateway);
    ::net::HttpServer server({.address = "127.0.0.1", .port = 0, .io_threads = 1});
    server.SetHttpRequestHandler([&adapter](std::shared_ptr<::net::IHttpRequest> request) {
        adapter.HandleHttp(std::move(request));
    });
    ASSERT_TRUE(server.Start().ok());

    auto create = SendJsonRequest(
        server.port(),
        ::net::http::verb::post,
        "/api/session/create",
        Json{
            {"traceId", "trace-http-create"},
            {"sessionId", "session-http"},
            {"userUuid", "user-http"},
            {"personaId", "dazhi"},
            {"personality", {{"name", "dazhi"}, {"description", "student"}}},
        });
    EXPECT_EQ(create.result(), ::net::http::status::ok);
    auto create_body = Json::parse(create.body());
    EXPECT_TRUE(create_body["ok"].get<bool>());
    EXPECT_EQ(create_body["data"]["sessionId"], "session-http");

    auto chat = SendJsonRequest(
        server.port(),
        ::net::http::verb::post,
        "/api/chat/message",
        Json{
            {"traceId", "trace-http-chat"},
            {"sessionId", "session-http"},
            {"personaId", "dazhi"},
            {"message", "hello"},
        });
    EXPECT_EQ(chat.result(), ::net::http::status::ok);
    auto chat_body = Json::parse(chat.body());
    EXPECT_TRUE(chat_body["ok"].get<bool>());
    EXPECT_EQ(chat_body["data"]["reply"]["content"], "student reply");
    EXPECT_EQ(chat_body["data"]["turnIndex"], 1);
    server.Stop();
}

TEST(PersonaGatewayHttpAdapterTest, PassesEmotionPromptsIntoSessionPromptBuilder) {
    GatewayFixture f;
    PersonaGatewayHttpAdapter adapter(f.gateway);
    ::net::HttpServer server({.address = "127.0.0.1", .port = 0, .io_threads = 1});
    server.SetHttpRequestHandler([&adapter](std::shared_ptr<::net::IHttpRequest> request) {
        adapter.HandleHttp(std::move(request));
    });
    ASSERT_TRUE(server.Start().ok());

    auto create = SendJsonRequest(
        server.port(),
        ::net::http::verb::post,
        "/api/session/create",
        Json{
            {"traceId", "trace-emotion-prompt-create"},
            {"sessionId", "session-emotion-prompt"},
            {"userUuid", "user-emotion-prompt"},
            {"personaId", "dazhi"},
            {"personality", {{"name", "dazhi"}, {"description", "student"}}},
            {"emotionPrompts", {
                {"emotionMap", {{"neutral", "用户状态平稳，保持自然教学节奏"}}},
                {"emotionReliability", {{"neutral", 1.0}}},
                {"confidenceThresholds", {{"strong", 0.5}, {"weak", 0.3}}},
                {"intensityLevels", {{"high_min", 0.7}}},
            }},
        });
    ASSERT_EQ(create.result(), ::net::http::status::ok);

    auto chat = SendJsonRequest(
        server.port(),
        ::net::http::verb::post,
        "/api/chat/message",
        Json{
            {"traceId", "trace-emotion-prompt-chat"},
            {"sessionId", "session-emotion-prompt"},
            {"personaId", "dazhi"},
            {"message", "hello"},
        });
    ASSERT_EQ(chat.result(), ::net::http::status::ok);
    {
        std::lock_guard lock(f.llm->mutex_);
        ASSERT_FALSE(f.llm->last_request.messages.empty());
        EXPECT_NE(f.llm->last_request.messages.front().content.find("用户状态平稳"), std::string::npos);
    }
    server.Stop();
}

TEST(PersonaGatewayHttpAdapterTest, AuthIdentityOverridesCreateSessionUserUuid) {
    GatewayFixture f;
    AuthIdentity identity;
    identity.authenticated = true;
    identity.user_uuid = "jwt-user-001";
    identity.tenant_id = "tenant-a";
    PersonaGatewayHttpAdapter adapter(f.gateway, std::make_shared<FixedAuthenticator>(identity));
    ::net::HttpServer server({.address = "127.0.0.1", .port = 0, .io_threads = 1});
    server.SetHttpRequestHandler([&adapter](std::shared_ptr<::net::IHttpRequest> request) {
        adapter.HandleHttp(std::move(request));
    });
    ASSERT_TRUE(server.Start().ok());

    auto create = SendJsonRequest(
        server.port(),
        ::net::http::verb::post,
        "/api/session/create",
        Json{
            {"traceId", "trace-auth-create"},
            {"sessionId", "session-auth"},
            {"userUuid", "forged-body-user"},
            {"personaId", "dazhi"},
            {"personality", {{"name", "dazhi"}, {"description", "student"}}},
        });
    EXPECT_EQ(create.result(), ::net::http::status::ok);
    auto create_body = Json::parse(create.body());
    EXPECT_EQ(create_body["data"]["userUuid"], "jwt-user-001");
    server.Stop();
}

TEST(PersonaGatewayHttpAdapterTest, RejectsApiWhenAuthenticatorRejects) {
    GatewayFixture f;
    PersonaGatewayHttpAdapter adapter(f.gateway, std::make_shared<RejectingAuthenticator>());
    ::net::HttpServer server({.address = "127.0.0.1", .port = 0, .io_threads = 1});
    server.SetHttpRequestHandler([&adapter](std::shared_ptr<::net::IHttpRequest> request) {
        adapter.HandleHttp(std::move(request));
    });
    ASSERT_TRUE(server.Start().ok());

    auto create = SendJsonRequest(
        server.port(),
        ::net::http::verb::post,
        "/api/session/create",
        Json{
            {"traceId", "trace-auth-reject"},
            {"sessionId", "session-auth-reject"},
            {"personaId", "dazhi"},
            {"personality", {{"name", "dazhi"}, {"description", "student"}}},
        });
    EXPECT_EQ(create.result(), ::net::http::status::forbidden);
    auto body = Json::parse(create.body());
    EXPECT_FALSE(body["ok"].get<bool>());
    EXPECT_EQ(body["error"]["code"], "PERMISSION_DENIED");
    server.Stop();
}

TEST(PersonaGatewayHttpAdapterTest, HealthRouteBypassesAuthenticatorAndReportsReadiness) {
    GatewayFixture f;
    PersonaGatewayHttpAdapter adapter(f.gateway, std::make_shared<RejectingAuthenticator>());
    ::net::HttpServer server({.address = "127.0.0.1", .port = 0, .io_threads = 1});
    server.SetHttpRequestHandler([&adapter](std::shared_ptr<::net::IHttpRequest> request) {
        adapter.HandleHttp(std::move(request));
    });
    ASSERT_TRUE(server.Start().ok());

    auto response = SendJsonRequest(
        server.port(),
        ::net::http::verb::get,
        "/api/health",
        Json{{"traceId", "trace-health"}});
    EXPECT_EQ(response.result(), ::net::http::status::ok);
    EXPECT_EQ(std::string(response["X-Trace-Id"]), "trace-health");
    auto body = Json::parse(response.body());
    EXPECT_TRUE(body["ok"].get<bool>());
    EXPECT_EQ(body["traceId"], "trace-health");
    EXPECT_EQ(body["data"]["status"], "ok");
    EXPECT_TRUE(body["data"].contains("sessionCount"));
    EXPECT_TRUE(body["data"]["pools"].contains("compute"));
    EXPECT_TRUE(body["data"]["pools"].contains("io"));
    server.Stop();
}

TEST(PersonaGatewayHttpAdapterTest, RegisterRouteBypassesAuthenticatorAndSetsCookie) {
    GatewayFixture f;
    auto registration = std::make_shared<FixedAuthRegistrationService>();
    PersonaGatewayHttpAdapter adapter(f.gateway, std::make_shared<RejectingAuthenticator>(), registration);
    ::net::HttpServer server({.address = "127.0.0.1", .port = 0, .io_threads = 1});
    server.SetHttpRequestHandler([&adapter](std::shared_ptr<::net::IHttpRequest> request) {
        adapter.HandleHttp(std::move(request));
    });
    ASSERT_TRUE(server.Start().ok());

    auto response = SendJsonRequest(
        server.port(),
        ::net::http::verb::post,
        "/api/auth/register",
        Json{
            {"traceId", "trace-register"},
            {"userUuid", "e2e-user-001"},
            {"tenantId", "default"},
            {"subject", "student@example.test"},
            {"ttlSeconds", 600},
        });
    EXPECT_EQ(response.result(), ::net::http::status::ok);
    EXPECT_NE(std::string(response[::net::http::field::set_cookie]).find("agent_auth=jwt-token"), std::string::npos);
    auto body = Json::parse(response.body());
    EXPECT_TRUE(body["ok"].get<bool>());
    EXPECT_EQ(body["data"]["authenticated"], true);
    EXPECT_EQ(body["data"]["userUuid"], "e2e-user-001");
    EXPECT_EQ(body["data"]["tenantId"], "default");
    EXPECT_EQ(body["data"]["subject"], "student@example.test");
    EXPECT_EQ(body["data"]["token"], "jwt-token");
    EXPECT_EQ(registration->last_request.ttl.count(), 600);
    server.Stop();
}

TEST(GatewayAuthSessionStoreTest, PersistsResolvesAndRevokesSessions) {
    const auto path = std::filesystem::temp_directory_path() / "agent_gateway_auth_sessions_test.db";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + "-wal", ec);
    std::filesystem::remove(path.string() + "-shm", ec);

    SqliteAuthSessionStore store(path.string());
    ASSERT_TRUE(store.EnsureSchema().ok());

    AuthSessionRecord record;
    record.token_id = "token-001";
    record.user_uuid = "uuid-001";
    record.tenant_id = "tenant-a";
    record.subject = "subject-001";
    record.issued_at = std::chrono::system_clock::now();
    record.expires_at = record.issued_at + std::chrono::hours(1);
    ASSERT_TRUE(store.UpsertSession(record).ok());

    auto resolved = store.ResolveSession("token-001");
    ASSERT_TRUE(resolved.ok()) << resolved.status().message();
    EXPECT_EQ(resolved.value().user_uuid, "uuid-001");
    EXPECT_EQ(resolved.value().tenant_id, "tenant-a");
    EXPECT_FALSE(resolved.value().revoked);

    ASSERT_TRUE(store.RevokeSession("token-001", "logout").ok());
    auto revoked = store.ResolveSession("token-001");
    ASSERT_TRUE(revoked.ok()) << revoked.status().message();
    EXPECT_TRUE(revoked.value().revoked);

    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + "-wal", ec);
    std::filesystem::remove(path.string() + "-shm", ec);
}

TEST(GatewayAuthSessionStoreTest, RedisPersistsResolvesAndRevokesSessionsWhenAvailable) {
    agent::semantic_cache::RedisPoolOptions options;
    options.host = "127.0.0.1";
    options.port = "5000";
    options.pool_size = 4;
    options.connect_timeout = std::chrono::seconds(1);
    options.command_timeout = std::chrono::milliseconds(1000);
    auto redis = std::make_shared<agent::semantic_cache::RedisConnectionPool>(options);
    auto started = redis->Start();
    if (!started.ok()) {
        GTEST_SKIP() << started.message();
    }

    RedisAuthSessionStore store(redis, "agent:test:gateway:auth");
    ASSERT_TRUE(store.EnsureSchema().ok());

    AuthSessionRecord record;
    record.token_id = "redis-token-001";
    record.user_uuid = "redis-uuid-001";
    record.tenant_id = "tenant-a";
    record.subject = "subject-001";
    record.issued_at = std::chrono::system_clock::now();
    record.expires_at = record.issued_at + std::chrono::hours(1);
    ASSERT_TRUE(store.UpsertSession(record).ok());

    auto resolved = store.ResolveSession("redis-token-001");
    ASSERT_TRUE(resolved.ok()) << resolved.status().message();
    EXPECT_EQ(resolved.value().user_uuid, "redis-uuid-001");
    EXPECT_EQ(resolved.value().tenant_id, "tenant-a");
    EXPECT_FALSE(resolved.value().revoked);

    ASSERT_TRUE(store.RevokeSession("redis-token-001", "logout").ok());
    auto revoked = store.ResolveSession("redis-token-001");
    ASSERT_TRUE(revoked.ok()) << revoked.status().message();
    EXPECT_TRUE(revoked.value().revoked);

    auto del = redis->Del({"agent:test:gateway:auth:session:redis-token-001"});
    EXPECT_TRUE(del.ok()) << del.status().message();
    redis->Shutdown();
}

TEST(PersonaGatewayServerTest, HostsApiWebSocketAdapterAndStaticDistOnOneHttpServer) {
    const auto static_root = std::filesystem::current_path() / "persona_gateway_static_test";
    std::filesystem::create_directories(static_root);
    {
        std::ofstream index(static_root / "index.html", std::ios::trunc);
        index << "<!doctype html><title>Gateway Dist</title>";
    }

    auto cache = std::make_shared<FakeSemanticCache>();
    auto memory = std::make_shared<SemanticMemoryContextProvider>(cache);
    auto emotion = std::make_shared<NeutralEmotionAnalyzer>();
    auto llm = std::make_shared<FakeLlmClient>();

    PersonaGatewayServerOptions options;
    options.http.address = "127.0.0.1";
    options.http.port = 0;
    options.http.io_threads = 1;
    options.compute_pool.worker_count = 1;
    options.compute_pool.queue_capacity = 64;
    options.io_pool.worker_count = 1;
    options.io_pool.queue_capacity = 64;
    options.runtime.default_model = "test-model";
    options.static_files = ::net::StaticFileOptions{.root = static_root, .index_file = "index.html", .spa_fallback = true};

    PersonaGatewayServerDependencies dependencies;
    dependencies.memory_provider = memory;
    dependencies.emotion_analyzer = emotion;
    dependencies.llm_client = llm;
   {
    PersonaGatewayServer server(std::move(options), std::move(dependencies));
    ASSERT_TRUE(server.Start().ok());

    auto create = SendJsonRequest(
        server.port(),
        ::net::http::verb::post,
        "/api/session/create",
        Json{
            {"traceId", "trace-server-create"},
            {"sessionId", "session-server"},
            {"userUuid", "user-server"},
            {"personaId", "dazhi"},
            {"personality", {{"name", "dazhi"}, {"description", "student"}}},
        });
    EXPECT_EQ(create.result(), ::net::http::status::ok);
    EXPECT_TRUE(Json::parse(create.body())["ok"].get<bool>());

    auto index = SendJsonRequest(server.port(), ::net::http::verb::get, "/", Json::object());
    EXPECT_EQ(index.result(), ::net::http::status::ok);
    EXPECT_NE(index.body().find("Gateway Dist"), std::string::npos);

    server.Stop();
   }
    std::error_code cleanup_error;
    std::filesystem::remove_all(static_root, cleanup_error);
}

} // namespace

