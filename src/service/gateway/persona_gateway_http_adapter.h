#pragma once

#include "gateway_auth.h"
#include "persona_gateway_service.h"
#include "request_interfaces.h"

#include <memory>
#include <string_view>

namespace agent::service::gateway {

class PersonaGatewayHttpAdapter {
public:
    explicit PersonaGatewayHttpAdapter(PersonaGatewayService& service,
                                       std::shared_ptr<IGatewayAuthenticator> authenticator = nullptr,
                                       std::shared_ptr<IAuthRegistrationService> auth_registration = nullptr);

    static bool IsApiRequest(std::string_view target) noexcept;

    void HandleHttp(std::shared_ptr<::net::IHttpRequest> request);
    void HandleWebSocket(std::shared_ptr<::net::IWebSocketStreamRequest> request);

private:
    PersonaGatewayService& service_;
    std::shared_ptr<IGatewayAuthenticator> authenticator_;
    std::shared_ptr<IAuthRegistrationService> auth_registration_;
};

} // namespace agent::service::gateway

