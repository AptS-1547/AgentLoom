#pragma once

#include "skill_session_manager.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace agent::service::persona {

// 静态注册阶段只保存身份和工厂，不创建线程、连接或其它运行时资源。
struct StatefulSkillExecutionContext {
    std::shared_ptr<ISkillSessionManager> sessions;
    SkillSessionStartRequest start_request;
};

class IStatefulSkillExecution {
public:
    virtual ~IStatefulSkillExecution() = default;
    virtual core::Status Start() = 0;
    virtual core::Status Stop(const SkillSessionStopRequest& request) = 0;
};

class StatefulSkillRegistry {
public:
    using Factory = std::function<std::shared_ptr<IStatefulSkillExecution>(
        StatefulSkillExecutionContext)>;

    static StatefulSkillRegistry& Instance() {
        static StatefulSkillRegistry registry;
        return registry;
    }

    core::Status Register(std::string skill_id, std::string version, Factory factory) {
        if (skill_id.empty() || version.empty() || !factory) {
            return core::Status::Error(core::ErrorCode::InvalidArgument,
                                       "stateful skill id, version and factory are required");
        }
        std::lock_guard lock(mutex_);
        if (factories_.contains(skill_id)) {
            registration_status_ = core::Status::Error(
                core::ErrorCode::AlreadyExists,
                "stateful skill factory already registered: " + skill_id);
            return registration_status_;
        }
        factories_.emplace(std::move(skill_id), Entry{std::move(version), std::move(factory)});
        return core::Status::Ok();
    }

    core::Result<Factory> Resolve(std::string_view skill_id,
                                  std::string_view version = {}) const {
        std::lock_guard lock(mutex_);
        auto it = factories_.find(std::string(skill_id));
        if (it == factories_.end()) {
            return core::Status::Error(core::ErrorCode::NotFound,
                                       "stateful skill factory not found");
        }
        if (!version.empty() && it->second.version != version) {
            return core::Status::Error(core::ErrorCode::NotFound,
                                       "stateful skill factory version not found");
        }
        return it->second.factory;
    }

    bool Contains(std::string_view skill_id) const {
        std::lock_guard lock(mutex_);
        return factories_.contains(std::string(skill_id));
    }

    core::Status ValidateRegistrations() const {
        std::lock_guard lock(mutex_);
        return registration_status_;
    }

private:
    struct Entry {
        std::string version;
        Factory factory;
    };

    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> factories_;
    core::Status registration_status_ = core::Status::Ok();
};

// 将通用 Session 生命周期与具体的有状态 Skill execution 连接起来。
class IStatefulSkillExecutionRouter {
public:
    virtual ~IStatefulSkillExecutionRouter() = default;
    virtual core::Result<SkillSessionSnapshot> Start(
        SkillSessionStartRequest request, std::string_view version = {}) = 0;
    virtual core::Result<SkillSessionSnapshot> Stop(
        const SkillSessionStopRequest& request) = 0;
};

class StatefulSkillExecutionRouter final : public IStatefulSkillExecutionRouter {
public:
    explicit StatefulSkillExecutionRouter(std::shared_ptr<ISkillSessionManager> sessions)
        : sessions_(std::move(sessions)) {}

    core::Result<SkillSessionSnapshot> Start(
        SkillSessionStartRequest request, std::string_view version = {}) override {
        if (!sessions_) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                       "skill session manager is not configured");
        }
        {
            std::lock_guard lock(mutex_);
            if (active_.contains(Key(request.session_id, request.skill_id))) {
                auto current = sessions_->Get(request.session_id, request.skill_id);
                if (!current.ok()) return current.status();
                if (current.value().has_value()) return *current.value();
            }
        }
        auto factory = StatefulSkillRegistry::Instance().Resolve(request.skill_id, version);
        if (!factory.ok() && factory.status().code() != core::ErrorCode::NotFound) {
            return factory.status();
        }
        auto started = sessions_->Start(request);
        if (!started.ok()) return started.status();
        if (!factory.ok()) return started;

        request.execution_id = started.value().execution_id;
        auto execution = factory.value()(StatefulSkillExecutionContext{sessions_, request});
        if (!execution) {
            sessions_->MarkFailed(request.session_id, request.skill_id,
                                  "stateful skill factory returned null",
                                  request.trace_id, request.execution_id);
            return core::Status::Error(core::ErrorCode::InternalError,
                                       "stateful skill factory returned null");
        }
        auto status = execution->Start();
        if (!status.ok()) {
            sessions_->MarkFailed(request.session_id, request.skill_id,
                                  status.message(), request.trace_id, request.execution_id);
            return status;
        }
        std::lock_guard lock(mutex_);
        active_[Key(request.session_id, request.skill_id)] = std::move(execution);
        auto current = sessions_->Get(request.session_id, request.skill_id);
        if (!current.ok()) return current.status();
        if (current.value().has_value()) return std::move(*current.value());
        return started;
    }

    core::Result<SkillSessionSnapshot> Stop(
        const SkillSessionStopRequest& request) override {
        if (!sessions_) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                       "skill session manager is not configured");
        }
        std::shared_ptr<IStatefulSkillExecution> execution;
        {
            std::lock_guard lock(mutex_);
            auto it = active_.find(Key(request.session_id, request.skill_id));
            if (it != active_.end()) execution = it->second;
        }
        if (execution) {
            auto status = execution->Stop(request);
            if (!status.ok()) return status;
            std::lock_guard lock(mutex_);
            active_.erase(Key(request.session_id, request.skill_id));
            auto current = sessions_->Get(request.session_id, request.skill_id);
            if (!current.ok()) return current.status();
            if (current.value().has_value()) return *current.value();
        }
        return sessions_->Stop(request);
    }

private:
    static std::string Key(std::string_view session_id, std::string_view skill_id) {
        return std::string(session_id) + "\n" + std::string(skill_id);
    }

    std::shared_ptr<ISkillSessionManager> sessions_;
    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<IStatefulSkillExecution>> active_;
};

template <typename T>
class StatefulSkillRegistrar {
public:
    StatefulSkillRegistrar(std::string_view skill_id, std::string_view version) {
        StatefulSkillRegistry::Instance().Register(
            std::string(skill_id), std::string(version),
            [](StatefulSkillExecutionContext context) {
                return std::make_shared<T>(std::move(context));
            });
    }
};

} // namespace agent::service::persona

// 使用唯一类名生成静态注册对象；宏不持有运行时资源。
#define REGISTER_STATEFUL_SKILL(ClassName, SkillId, Version) \
    static const ::agent::service::persona::StatefulSkillRegistrar<ClassName> \
        g_stateful_skill_registrar_##ClassName{SkillId, Version}
