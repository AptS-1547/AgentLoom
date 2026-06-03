#pragma once

#include "persona_gateway_service.h"
#include "request_interfaces.h"

#include <memory>
#include <string_view>

namespace agent::service::gateway {

class PersonaGatewayHttpAdapter {
public:
    explicit PersonaGatewayHttpAdapter(PersonaGatewayService& service);

    static bool IsApiRequest(std::string_view target) noexcept;

    void HandleHttp(std::shared_ptr<::net::IHttpRequest> request);
    void HandleWebSocket(std::shared_ptr<::net::IWebSocketStreamRequest> request);

private:
    PersonaGatewayService& service_;
};

} // namespace agent::service::gateway

