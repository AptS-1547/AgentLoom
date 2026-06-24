#pragma once

#include "gateway_auth.h"
#include "document_analysis_service.h"
#include "isemantic_cache.h"
#include "persona_gateway_service.h"
#include "request_interfaces.h"
#include "skill_session_manager.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>

namespace agent::service::gateway {

class PersonaGatewayHttpAdapter {
public:
    struct DocumentUploadSession {
        std::string upload_id;
        std::string file_name;
        std::string owner_user_uuid;
        std::string session_id;
        std::filesystem::path temp_path;
        std::uint64_t expected_size = 0;
        std::uint64_t received_size = 0;
        std::uint64_t connection_id = 0;
        bool binary_mode = false;
    };

    explicit PersonaGatewayHttpAdapter(PersonaGatewayService& service,
                                       std::shared_ptr<IGatewayAuthenticator> authenticator = nullptr,
                                       std::shared_ptr<IAuthRegistrationService> auth_registration = nullptr,
                                       std::shared_ptr<document::DocumentAnalysisService> document_service = nullptr,
                                       std::shared_ptr<llm::ILlmClient> llm_client = nullptr,
                                       std::shared_ptr<document::IDocumentEmbeddingProvider> embedding_provider = nullptr,
                                       std::shared_ptr<document::IDocumentLlmChunkCache> llm_chunk_cache = nullptr,
                                       std::shared_ptr<semantic_cache::ISemanticCache> document_semantic_cache = nullptr,
                                       std::shared_ptr<persona::ISkillSessionManager> skill_session_manager = nullptr);

    static bool IsApiRequest(std::string_view target) noexcept;

    void HandleHttp(std::shared_ptr<::net::IHttpRequest> request);
    void HandleWebSocket(std::shared_ptr<::net::IWebSocketStreamRequest> request);

private:
    PersonaGatewayService& service_;
    std::shared_ptr<IGatewayAuthenticator> authenticator_;
    std::shared_ptr<IAuthRegistrationService> auth_registration_;
    std::shared_ptr<document::DocumentAnalysisService> document_service_;
    std::shared_ptr<llm::ILlmClient> llm_client_;
    std::shared_ptr<document::IDocumentEmbeddingProvider> embedding_provider_;
    std::shared_ptr<document::IDocumentLlmChunkCache> llm_chunk_cache_;
    std::shared_ptr<semantic_cache::ISemanticCache> document_semantic_cache_;
    std::shared_ptr<persona::ISkillSessionManager> skill_session_manager_;
    std::mutex document_upload_mutex_;
    std::unordered_map<std::string, DocumentUploadSession> document_uploads_;
};

} // namespace agent::service::gateway

