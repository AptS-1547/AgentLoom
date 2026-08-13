#include <AgentLoom/core/memory_pool.h>
#include <AgentLoom/config/config_section.h>
#include <AgentLoom/config/server_config.h>
#include <AgentLoom/generated/bert_inference.grpc.pb.h>
#include <AgentLoom/service/gateway/gateway_lifecycle.h>
#include <AgentLoom/service/gateway/gateway_routing.h>
#include <AgentLoom/service/persona/persona_interaction.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct ConsumerRouteContext {};

class ConsumerRoute final
    : public agent::service::gateway::ITypedHttpRoute<ConsumerRouteContext> {
public:
    ::net::http::verb Method() const noexcept override {
        return ::net::http::verb::get;
    }

    std::vector<std::string_view> Pattern() const override {
        return {"runtime", "{runtime_id}"};
    }

    void Handle(ConsumerRouteContext&) const override {}
};

std::unique_ptr<agent::service::gateway::ITypedHttpRoute<ConsumerRouteContext>>
MakeConsumerRoute() {
    return std::make_unique<ConsumerRoute>();
}

}

int main() {
    core::BucketMemoryPool pool;
    auto block = pool.allocate(256);
    if (!block.ok() || block.value().size() != 256) {
        return 1;
    }

    server_config::Json helper_json{{"value", "from-helper"}};
    std::string helper_value;
    server_config::SetString(helper_json, "helper", "value", helper_value);
    if (helper_value != "from-helper") {
        return 2;
    }

    const auto selection = server_config::ConfigSectionSelection::Only({"llm", "sdk_consumer"});
    const auto sections = server_config::BuildConfigSections(selection);
    if (sections.size() != 2) {
        return 3;
    }

    const auto temp = std::filesystem::temp_directory_path() / "agentloom_package_consumer";
    std::filesystem::create_directories(temp);
    {
        std::ofstream key(temp / "api-key.txt", std::ios::binary);
        key << "package-key\n";
        std::ofstream config(temp / "config.json", std::ios::binary);
        config << R"({
            "llm": {
                "enabled": true,
                "base_url": "https://example.invalid/v1",
                "model": "package-model",
                "api_key_file": "api-key.txt"
            },
            "sdk_consumer": { "value": "registered" }
        })";
    }

    MultimodalServerOptions options;
    server_config::LoadConfigFile(temp / "config.json", options, selection);
    server_config::ValidateOptions(options, selection);
    std::error_code ignored;
    std::filesystem::remove_all(temp, ignored);

    bert_inference::PredictRequest generated_request;
    generated_request.add_input_ids(42);
    agent::service::gateway::TypedRouteRegistry<
        agent::service::gateway::ITypedHttpRoute<ConsumerRouteContext>> route_registry;
    if (!route_registry.Register("consumer-runtime", &MakeConsumerRoute).ok()) {
        return 5;
    }
    agent::service::gateway::TypedHttpRouteDispatcher<ConsumerRouteContext> dispatcher(route_registry);
    auto route = dispatcher.Match(::net::http::verb::get, {"runtime", "runtime-001"});
    if (!route.ok() || route.value().path_params.at("runtime_id") != "runtime-001") {
        return 6;
    }
    agent::service::gateway::GatewayLifecycleCoordinator lifecycle;
    if (!lifecycle.Start().ok() || !lifecycle.Stop().ok()) {
        return 7;
    }
    return options.llm.api_key == "package-key" &&
                   options.auth.token == "registered" &&
                   generated_request.input_ids_size() == 1 &&
                   generated_request.input_ids(0) == 42
               ? 0
               : 4;
}
