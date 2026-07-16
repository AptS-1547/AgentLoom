#pragma once

#include "document_analysis_service.h"
#include "gateway_auth.h"
#include "isemantic_cache.h"
#include "persona_gateway_service.h"
#include "request_interfaces.h"
#include "skill_session_manager.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace agent::service::gateway {

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

struct HttpRouteContext {
    PersonaGatewayService& service;
    std::shared_ptr<document::DocumentAnalysisService> document_service;
    std::shared_ptr<llm::ILlmClient> llm_client;
    std::shared_ptr<document::IDocumentEmbeddingProvider> embedding_provider;
    std::shared_ptr<document::IDocumentLlmChunkCache> llm_chunk_cache;
    std::shared_ptr<semantic_cache::ISemanticCache> document_semantic_cache;
    std::shared_ptr<persona::ISkillSessionManager> skill_session_manager;
    std::shared_ptr<IAuthRegistrationService> auth_registration;
    bool enable_dev_registration = false;
    bool enable_path_register_test_endpoint = false;
    bool enable_path_analyze_test_endpoint = false;
    std::shared_ptr<::net::IHttpRequest> request;
    const ::net::BeastHttpRequest& message;
    const nlohmann::json& body;
    const AuthIdentity& identity;
    std::string trace_id;
    std::vector<std::string> path_parts;
    std::unordered_map<std::string, std::string> path_params;
};

struct WsRouteContext {
    PersonaGatewayService& service;
    std::shared_ptr<document::DocumentAnalysisService> document_service;
    std::shared_ptr<llm::ILlmClient> llm_client;
    std::shared_ptr<document::IDocumentEmbeddingProvider> embedding_provider;
    std::shared_ptr<document::IDocumentLlmChunkCache> llm_chunk_cache;
    std::shared_ptr<semantic_cache::ISemanticCache> document_semantic_cache;
    std::shared_ptr<persona::ISkillSessionManager> skill_session_manager;
    std::shared_ptr<::net::IWebSocketStreamRequest> request;
    const nlohmann::json& body;
    const AuthIdentity& identity;
    std::string trace_id;
    std::mutex& upload_mutex;
    std::unordered_map<std::string, DocumentUploadSession>& document_uploads;
};

class IHttpRoute {
public:
    virtual ~IHttpRoute() = default;
    virtual ::net::http::verb Method() const noexcept = 0;
    virtual std::vector<std::string_view> Pattern() const = 0;
    virtual bool RequiresAuth() const noexcept { return true; }
    virtual bool RequiresAuthenticatedIdentity() const noexcept { return false; }
    virtual void Handle(HttpRouteContext& context) const = 0;

    bool Matches(::net::http::verb method,
                 const std::vector<std::string>& parts,
                 std::unordered_map<std::string, std::string>& params) const {
        if (method != Method()) {
            return false;
        }
        const auto pattern = Pattern();
        if (pattern.size() != parts.size()) {
            return false;
        }
        params.clear();
        for (std::size_t i = 0; i < pattern.size(); ++i) {
            const auto token = pattern[i];
            if (token.size() >= 2 && token.front() == '{' && token.back() == '}') {
                params.emplace(std::string(token.substr(1, token.size() - 2)), parts[i]);
                continue;
            }
            if (token != parts[i]) {
                return false;
            }
        }
        return true;
    }
};

using HttpRouteFactory = std::unique_ptr<IHttpRoute> (*)();

class IWsRoute {
public:
    virtual ~IWsRoute() = default;
    virtual std::string_view Type() const noexcept = 0;
    virtual bool RequiresAuth() const noexcept { return true; }
    virtual bool RequiresAuthenticatedIdentity() const noexcept { return false; }
    virtual void Handle(WsRouteContext& context) const = 0;

    bool Matches(std::string_view type) const noexcept {
        return Type() == type;
    }
};

using WsRouteFactory = std::unique_ptr<IWsRoute> (*)();

class HttpRouteRegistry {
public:
    static HttpRouteRegistry& Instance() {
        static HttpRouteRegistry registry;
        return registry;
    }

    bool Register(std::string_view name, HttpRouteFactory factory) {
        auto duplicate = std::find_if(
            entries_.begin(),
            entries_.end(),
            [name](const Entry& entry) {
                return entry.name == name;
            });
        if (duplicate == entries_.end()) {
            entries_.push_back({name, factory});
        }
        return true;
    }

    std::vector<std::unique_ptr<IHttpRoute>> CreateRoutes() const {
        std::vector<std::unique_ptr<IHttpRoute>> routes;
        routes.reserve(entries_.size());
        for (const auto& entry : entries_) {
            routes.push_back(entry.factory());
        }
        return routes;
    }

private:
    struct Entry {
        std::string_view name;
        HttpRouteFactory factory = nullptr;
    };

    std::vector<Entry> entries_;
};

class WsRouteRegistry {
public:
    static WsRouteRegistry& Instance() {
        static WsRouteRegistry registry;
        return registry;
    }

    bool Register(std::string_view name, WsRouteFactory factory) {
        auto duplicate = std::find_if(
            entries_.begin(),
            entries_.end(),
            [name](const Entry& entry) {
                return entry.name == name;
            });
        if (duplicate == entries_.end()) {
            entries_.push_back({name, factory});
        }
        return true;
    }

    std::vector<std::unique_ptr<IWsRoute>> CreateRoutes() const {
        std::vector<std::unique_ptr<IWsRoute>> routes;
        routes.reserve(entries_.size());
        for (const auto& entry : entries_) {
            routes.push_back(entry.factory());
        }
        return routes;
    }

private:
    struct Entry {
        std::string_view name;
        WsRouteFactory factory = nullptr;
    };

    std::vector<Entry> entries_;
};

template <typename T>
class HttpRouteRegistrar {
public:
    HttpRouteRegistrar() {
        HttpRouteRegistry::Instance().Register(
            T::kRouteName,
            []() -> std::unique_ptr<IHttpRoute> {
                return std::make_unique<T>();
            });
    }
};

template <typename T>
class WsRouteRegistrar {
public:
    WsRouteRegistrar() {
        WsRouteRegistry::Instance().Register(
            T::kRouteName,
            []() -> std::unique_ptr<IWsRoute> {
                return std::make_unique<T>();
            });
    }
};

#define DECLARE_HTTP_ROUTE(ClassName, MethodValue, ...) \
class ClassName final : public IHttpRoute { \
public: \
    static constexpr std::string_view kRouteName = #ClassName; \
    ::net::http::verb Method() const noexcept override { return MethodValue; } \
    std::vector<std::string_view> Pattern() const override { return {__VA_ARGS__}; } \
    void Handle(HttpRouteContext& context) const override; \
}; \
static const HttpRouteRegistrar<ClassName> g_##ClassName##_registrar; \
void ClassName::Handle(HttpRouteContext& context) const

#define DECLARE_AUTHENTICATED_HTTP_ROUTE(ClassName, MethodValue, ...) \
class ClassName final : public IHttpRoute { \
public: \
    static constexpr std::string_view kRouteName = #ClassName; \
    ::net::http::verb Method() const noexcept override { return MethodValue; } \
    std::vector<std::string_view> Pattern() const override { return {__VA_ARGS__}; } \
    bool RequiresAuthenticatedIdentity() const noexcept override { return true; } \
    void Handle(HttpRouteContext& context) const override; \
}; \
static const HttpRouteRegistrar<ClassName> g_##ClassName##_registrar; \
void ClassName::Handle(HttpRouteContext& context) const

#define DECLARE_PUBLIC_HTTP_ROUTE(ClassName, MethodValue, ...) \
class ClassName final : public IHttpRoute { \
public: \
    static constexpr std::string_view kRouteName = #ClassName; \
    ::net::http::verb Method() const noexcept override { return MethodValue; } \
    std::vector<std::string_view> Pattern() const override { return {__VA_ARGS__}; } \
    bool RequiresAuth() const noexcept override { return false; } \
    void Handle(HttpRouteContext& context) const override; \
}; \
static const HttpRouteRegistrar<ClassName> g_##ClassName##_registrar; \
void ClassName::Handle(HttpRouteContext& context) const

#define DECLARE_WS_ROUTE(ClassName, TypeValue) \
class ClassName final : public IWsRoute { \
public: \
    static constexpr std::string_view kRouteName = #ClassName; \
    std::string_view Type() const noexcept override { return TypeValue; } \
    void Handle(WsRouteContext& context) const override; \
}; \
static const WsRouteRegistrar<ClassName> g_##ClassName##_registrar; \
void ClassName::Handle(WsRouteContext& context) const

#define DECLARE_AUTHENTICATED_WS_ROUTE(ClassName, TypeValue) \
class ClassName final : public IWsRoute { \
public: \
    static constexpr std::string_view kRouteName = #ClassName; \
    std::string_view Type() const noexcept override { return TypeValue; } \
    bool RequiresAuthenticatedIdentity() const noexcept override { return true; } \
    void Handle(WsRouteContext& context) const override; \
}; \
static const WsRouteRegistrar<ClassName> g_##ClassName##_registrar; \
void ClassName::Handle(WsRouteContext& context) const

} // namespace agent::service::gateway
