#include "gateway_lifecycle.h"
#include "gateway_routing.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using agent::service::gateway::CallbackGatewayIngressMiddleware;
using agent::service::gateway::CallbackGatewayLifecycleComponent;
using agent::service::gateway::GatewayLifecycleCoordinator;
using agent::service::gateway::GatewayLifecycleState;
using agent::service::gateway::ITypedHttpRoute;
using agent::service::gateway::StripRoutePrefix;
using agent::service::gateway::TypedHttpRouteDispatcher;
using agent::service::gateway::TypedRouteRegistry;

std::shared_ptr<CallbackGatewayLifecycleComponent> MakeLifecycleComponent(
    std::string name,
    std::vector<std::string>& events,
    bool fail_start = false) {
    const auto component_name = name;
    return std::make_shared<CallbackGatewayLifecycleComponent>(
        std::move(name),
        [&events, fail_start, component = component_name]() {
            events.push_back("start:" + component);
            return fail_start
                ? core::Status::Error(core::ErrorCode::Unavailable, "injected start failure")
                : core::Status::Ok();
        },
        [&events, component = component_name](std::chrono::steady_clock::time_point) {
            events.push_back("stop:" + component);
            return core::Status::Ok();
        });
}

TEST(GatewayLifecycleCoordinatorTest, StartsInOrderAndStopsInReverseOrder) {
    std::vector<std::string> events;
    GatewayLifecycleCoordinator lifecycle;
    ASSERT_TRUE(lifecycle.Register(MakeLifecycleComponent("storage", events)).ok());
    ASSERT_TRUE(lifecycle.Register(MakeLifecycleComponent("workers", events)).ok());
    ASSERT_TRUE(lifecycle.Register(MakeLifecycleComponent("ingress", events)).ok());

    ASSERT_TRUE(lifecycle.Start().ok());
    EXPECT_EQ(lifecycle.state(), GatewayLifecycleState::Running);
    EXPECT_EQ(lifecycle.started_component_count(), 3u);
    ASSERT_TRUE(lifecycle.Stop().ok());
    EXPECT_EQ(lifecycle.state(), GatewayLifecycleState::Stopped);
    EXPECT_EQ(events, (std::vector<std::string>{
                          "start:storage",
                          "start:workers",
                          "start:ingress",
                          "stop:ingress",
                          "stop:workers",
                          "stop:storage",
                      }));
}

TEST(GatewayLifecycleCoordinatorTest, CleansFailedComponentAndRollsBackStartedComponents) {
    std::vector<std::string> events;
    GatewayLifecycleCoordinator lifecycle;
    ASSERT_TRUE(lifecycle.Register(MakeLifecycleComponent("storage", events)).ok());
    ASSERT_TRUE(lifecycle.Register(MakeLifecycleComponent("workers", events, true)).ok());
    ASSERT_TRUE(lifecycle.Register(MakeLifecycleComponent("ingress", events)).ok());

    const auto status = lifecycle.Start();
    EXPECT_EQ(status.code(), core::ErrorCode::Unavailable);
    EXPECT_EQ(lifecycle.state(), GatewayLifecycleState::Stopped);
    EXPECT_EQ(lifecycle.started_component_count(), 0u);
    EXPECT_EQ(events, (std::vector<std::string>{
                          "start:storage",
                          "start:workers",
                          "stop:workers",
                          "stop:storage",
                      }));
}

struct ConsumerRouteContext {
    std::string selected_id;
};

class ConsumerRoute final : public ITypedHttpRoute<ConsumerRouteContext> {
public:
    ::net::http::verb Method() const noexcept override {
        return ::net::http::verb::post;
    }

    std::vector<std::string_view> Pattern() const override {
        return {"sessions", "{session_id}", "turns"};
    }

    void Handle(ConsumerRouteContext&) const override {}
};

std::unique_ptr<ITypedHttpRoute<ConsumerRouteContext>> MakeConsumerRoute() {
    return std::make_unique<ConsumerRoute>();
}

TEST(GatewayRoutingTest, ConsumerPrefixMiddlewareAndTypedRouteAreIndependentOfReferenceGateway) {
    struct IngressRequest {
        bool admitted = false;
    };
    CallbackGatewayIngressMiddleware<IngressRequest> middleware(
        [](IngressRequest& request) {
            request.admitted = true;
            return core::Status::Ok();
        });
    IngressRequest ingress;
    ASSERT_TRUE(middleware.BeforeDispatch(ingress).ok());
    EXPECT_TRUE(ingress.admitted);

    auto relative = StripRoutePrefix(
        {"application", "v1", "sessions", "session-a", "turns"},
        {"application", "v1"});
    ASSERT_TRUE(relative.ok()) << relative.status().message();

    TypedRouteRegistry<ITypedHttpRoute<ConsumerRouteContext>> registry;
    ASSERT_TRUE(registry.Register("consumer-turn", &MakeConsumerRoute).ok());
    TypedHttpRouteDispatcher<ConsumerRouteContext> dispatcher(registry);
    auto match = dispatcher.Match(::net::http::verb::post, relative.value());
    ASSERT_TRUE(match.ok()) << match.status().message();
    EXPECT_EQ(match.value().path_params.at("session_id"), "session-a");
}

}
