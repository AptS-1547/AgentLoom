#pragma once

#include "http_types.h"
#include "result.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace agent::service::gateway {

template <typename Request>
class IGatewayIngressMiddleware {
public:
    virtual ~IGatewayIngressMiddleware() = default;

    /// 在创建业务上下文和保留请求载荷前执行 admission、deadline、trace 与身份适配。
    virtual core::Status BeforeDispatch(Request& request) const = 0;
};

template <typename Request>
class CallbackGatewayIngressMiddleware final : public IGatewayIngressMiddleware<Request> {
public:
    using Callback = std::function<core::Status(Request&)>;

    explicit CallbackGatewayIngressMiddleware(Callback callback)
        : callback_(std::move(callback)) {}

    core::Status BeforeDispatch(Request& request) const override {
        if (!callback_) {
            return core::Status::Ok();
        }
        return callback_(request);
    }

private:
    Callback callback_;
};

inline core::Result<std::vector<std::string>> StripRoutePrefix(
    const std::vector<std::string>& path_parts,
    const std::vector<std::string>& prefix_parts) {
    if (path_parts.size() < prefix_parts.size() ||
        !std::equal(prefix_parts.begin(), prefix_parts.end(), path_parts.begin())) {
        return core::Status::Error(core::ErrorCode::NotFound,
                                   "route prefix does not match");
    }
    return std::vector<std::string>(path_parts.begin() + prefix_parts.size(),
                                    path_parts.end());
}

template <typename Context>
class ITypedHttpRoute {
public:
    virtual ~ITypedHttpRoute() = default;
    virtual ::net::http::verb Method() const noexcept = 0;
    virtual std::vector<std::string_view> Pattern() const = 0;
    virtual bool RequiresAuth() const noexcept { return true; }
    virtual bool RequiresAuthenticatedIdentity() const noexcept { return false; }
    virtual void Handle(Context& context) const = 0;

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

template <typename Context>
class ITypedWsRoute {
public:
    virtual ~ITypedWsRoute() = default;
    virtual std::string_view Type() const noexcept = 0;
    virtual bool RequiresAuth() const noexcept { return true; }
    virtual bool RequiresAuthenticatedIdentity() const noexcept { return false; }
    virtual void Handle(Context& context) const = 0;

    bool Matches(std::string_view type) const noexcept {
        return Type() == type;
    }
};

template <typename Route>
class TypedRouteRegistry {
public:
    using Factory = std::unique_ptr<Route> (*)();

    core::Status Register(std::string_view name, Factory factory) {
        if (name.empty() || !factory) {
            return core::Status::Error(core::ErrorCode::InvalidArgument,
                                       "route name and factory are required");
        }
        std::lock_guard lock(mutex_);
        const auto duplicate = std::find_if(
            entries_.begin(), entries_.end(),
            [&name](const Entry& entry) { return entry.name == name; });
        if (duplicate != entries_.end()) {
            return core::Status::Error(core::ErrorCode::AlreadyExists,
                                       "route already registered: " + std::string(name));
        }
        entries_.push_back(Entry{std::string(name), factory});
        return core::Status::Ok();
    }

    std::vector<std::unique_ptr<Route>> CreateRoutes() const {
        std::lock_guard lock(mutex_);
        std::vector<std::unique_ptr<Route>> routes;
        routes.reserve(entries_.size());
        for (const auto& entry : entries_) {
            routes.push_back(entry.factory());
        }
        return routes;
    }

private:
    struct Entry {
        std::string name;
        Factory factory = nullptr;
    };

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
};

template <typename Context>
struct TypedHttpRouteMatch {
    std::unique_ptr<ITypedHttpRoute<Context>> route;
    std::unordered_map<std::string, std::string> path_params;
};

template <typename Context>
class TypedHttpRouteDispatcher {
public:
    using Route = ITypedHttpRoute<Context>;

    explicit TypedHttpRouteDispatcher(const TypedRouteRegistry<Route>& registry)
        : registry_(registry) {}

    core::Result<TypedHttpRouteMatch<Context>> Match(
        ::net::http::verb method,
        const std::vector<std::string>& parts) const {
        auto routes = registry_.CreateRoutes();
        for (auto& route : routes) {
            std::unordered_map<std::string, std::string> params;
            if (route->Matches(method, parts, params)) {
                return TypedHttpRouteMatch<Context>{std::move(route), std::move(params)};
            }
        }
        return core::Status::Error(core::ErrorCode::NotFound, "route not found");
    }

private:
    const TypedRouteRegistry<Route>& registry_;
};

template <typename Context>
class TypedWsRouteDispatcher {
public:
    using Route = ITypedWsRoute<Context>;

    explicit TypedWsRouteDispatcher(const TypedRouteRegistry<Route>& registry)
        : registry_(registry) {}

    core::Result<std::unique_ptr<Route>> Match(std::string_view type) const {
        auto routes = registry_.CreateRoutes();
        for (auto& route : routes) {
            if (route->Type() == type) {
                return std::move(route);
            }
        }
        return core::Status::Error(core::ErrorCode::NotFound,
                                   "websocket route not found");
    }

private:
    const TypedRouteRegistry<Route>& registry_;
};

}
