#include "classroom_scheduler.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace agent::service::gateway {
namespace {

bool IsSet(std::chrono::steady_clock::time_point time) {
    return time.time_since_epoch().count() != 0;
}

std::vector<std::string> SortedSet(const std::unordered_set<std::string>& values) {
    std::vector<std::string> out(values.begin(), values.end());
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

ClassroomScheduler::ClassroomScheduler(ClassroomSchedulerOptions options, core::LoggerAdapter logger)
    : options_(std::move(options)),
      logger_(std::move(logger)) {}

void ClassroomScheduler::SetDecisionProvider(std::shared_ptr<IProactiveDecisionProvider> provider) {
    std::lock_guard lock(mutex_);
    decision_provider_ = std::move(provider);
}

core::Status ClassroomScheduler::RegisterPersona(ClassroomPersonaRegistration request) {
    if (request.classroom_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "classroom_id is required");
    }
    if (request.persona_id.empty() || request.session_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "persona_id and session_id are required");
    }
    {
        try{
            std::lock_guard lock(mutex_);
            auto& classroom = classrooms_[request.classroom_id];
            auto& persona = classroom.personas[request.persona_id];
            persona.persona_id = request.persona_id;
            persona.session_id = request.session_id;
            persona.context_ids = std::move(request.context_ids);
            persona.context_patterns = std::move(request.context_patterns);
            persona.config_source = std::move(request.config_source);
            persona.agent = request.agent;
            persona.proactive = request.proactive;
            persona.default_persona = request.default_persona;
            persona.started_at = std::chrono::steady_clock::now();
            persona.last_user_time = persona.started_at;
            persona.proactive_state = ProactiveState::Normal;
                    
            if (persona.default_persona || classroom.default_persona.empty()) {
                classroom.default_persona = persona.persona_id;
            }
            classroom.context_cache.clear();

            logger_.info("[classroom] registered classroom={} persona={} session={} default={}",
                        request.classroom_id,
                        persona.persona_id,
                        persona.session_id,
                        persona.default_persona);
        } catch (const std::exception& e) {
            logger_.error("[classroom] RegisterPersona failed: {}", e.what());
            return core::Status::Error(core::ErrorCode::InternalError,
                std::string("RegisterPersona failed: ") + e.what());
        }
    }
    return core::Status::Ok();
}

core::Status ClassroomScheduler::UnregisterPersona(std::string_view classroom_id, std::string_view persona_id) {
    std::lock_guard lock(mutex_);
    auto classroom_it = classrooms_.find(std::string(classroom_id));
    if (classroom_it == classrooms_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "classroom not found");
    }
    auto erased = classroom_it->second.personas.erase(std::string(persona_id));
    if (erased == 0) {
        return core::Status::Error(core::ErrorCode::NotFound, "persona not found");
    }
    classroom_it->second.context_cache.clear();
    if (classroom_it->second.default_persona == persona_id) {
        classroom_it->second.default_persona.clear();
        for (const auto& [id, persona] : classroom_it->second.personas) {
            if (persona.default_persona) {
                classroom_it->second.default_persona = id;
                break;
            }
        }
    }
    if (classroom_it->second.personas.empty()) {
        classrooms_.erase(classroom_it);
    }
    logger_.info("[classroom] unregistered classroom={} persona={}", classroom_id, persona_id);
    return core::Status::Ok();
}

core::Status ClassroomScheduler::UnregisterSession(std::string_view session_id) {
    std::lock_guard lock(mutex_);
    bool removed = false;
    for (auto classroom_it = classrooms_.begin(); classroom_it != classrooms_.end();) {
        auto& classroom = classroom_it->second;
        for (auto persona_it = classroom.personas.begin(); persona_it != classroom.personas.end();) {
            if (persona_it->second.session_id == session_id) {
                if (classroom.default_persona == persona_it->first) {
                    classroom.default_persona.clear();
                }
                persona_it = classroom.personas.erase(persona_it);
                removed = true;
            } else {
                ++persona_it;
            }
        }
        classroom.context_cache.clear();
        if (classroom.default_persona.empty()) {
            for (const auto& [id, persona] : classroom.personas) {
                if (persona.default_persona) {
                    classroom.default_persona = id;
                    break;
                }
            }
        }
        if (classroom.personas.empty()) {
            classroom_it = classrooms_.erase(classroom_it);
        } else {
            ++classroom_it;
        }
    }
    return removed ? core::Status::Ok()
                   : core::Status::Error(core::ErrorCode::NotFound, "session is not registered in classroom scheduler");
}

core::Result<ClassroomRouteResult> ClassroomScheduler::Resolve(const ClassroomRouteRequest& request) {
    std::lock_guard lock(mutex_);
    auto classroom_it = classrooms_.find(request.classroom_id);
    if (classroom_it == classrooms_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "classroom not found");
    }
    return ResolveLocked(classroom_it->second, classroom_it->first, request);
}

core::Status ClassroomScheduler::OnUserMessage(std::string_view classroom_id, std::string_view persona_id) {
    std::lock_guard lock(mutex_);
    auto classroom_it = classrooms_.find(std::string(classroom_id));
    if (classroom_it == classrooms_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "classroom not found");
    }
    auto persona_it = classroom_it->second.personas.find(std::string(persona_id));
    if (persona_it == classroom_it->second.personas.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "persona not found");
    }
    persona_it->second.last_user_time = std::chrono::steady_clock::now();
    if (persona_it->second.proactive_state == ProactiveState::WaitingResponse ||
        persona_it->second.proactive_state == ProactiveState::Dormant) {
        persona_it->second.proactive_state = ProactiveState::Normal;
    }
    return core::Status::Ok();
}

core::Status ClassroomScheduler::OnProactiveDelivered(std::string_view classroom_id, std::string_view persona_id) {
    std::lock_guard lock(mutex_);
    auto classroom_it = classrooms_.find(std::string(classroom_id));
    if (classroom_it == classrooms_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "classroom not found");
    }
    auto persona_it = classroom_it->second.personas.find(std::string(persona_id));
    if (persona_it == classroom_it->second.personas.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "persona not found");
    }
    const auto now = std::chrono::steady_clock::now();
    persona_it->second.last_proactive_time = now;
    persona_it->second.last_proactive_attempt_time = now;
    if (persona_it->second.proactive.enabled) {
        persona_it->second.proactive_state = ProactiveState::WaitingResponse;
    }
    return core::Status::Ok();
}

core::Result<ClassroomPollDecision> ClassroomScheduler::Poll(const ClassroomPollRequest& request) {
    if (request.classroom_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "classroom_id is required");
    }

    std::shared_ptr<IProactiveDecisionProvider> provider;
    ClassroomPollDecision decision;
    ProactiveDecisionRequest provider_request;
    ProactiveOptions proactive_options;

    {
        std::lock_guard lock(mutex_);
        auto classroom_it = classrooms_.find(request.classroom_id);
        if (classroom_it == classrooms_.end()) {
            return core::Status::Error(core::ErrorCode::NotFound, "classroom not found");
        }

        ClassroomRouteRequest route;
        route.classroom_id = request.classroom_id;
        route.context_id = request.context_id;
        route.persona_hint = request.persona_hint;
        auto routed = ResolveLocked(classroom_it->second, classroom_it->first, route);
        if (!routed.ok()) {
            return routed.status();
        }

        auto* persona = FindPersonaLocked(classroom_it->second, routed.value().persona_id, routed.value().session_id);
        if (!persona) {
            return core::Status::Error(core::ErrorCode::NotFound, "persona not found");
        }

        decision.classroom_id = classroom_it->first;
        decision.persona_id = persona->persona_id;
        decision.session_id = persona->session_id;
        decision.state = persona->proactive_state;

        const auto now = std::chrono::steady_clock::now();
        const auto idle = std::chrono::duration_cast<std::chrono::seconds>(now - persona->last_user_time);
        if (persona->agent.proactive_level == ProactiveLevel::Off) {
            return decision;
        }
        if (persona->proactive_state == ProactiveState::Dormant) {
            return decision;
        }
        if (persona->proactive_state == ProactiveState::WaitingResponse) {
            const auto waited = std::chrono::duration_cast<std::chrono::minutes>(
                now - persona->last_proactive_attempt_time);
            if (waited >= persona->proactive.response_wait) {
                persona->proactive_state = ProactiveState::Dormant;
                decision.state = persona->proactive_state;
            }
            return decision;
        }
        if (request.system_event) {
            if (IsSet(persona->last_proactive_time) &&
                now - persona->last_proactive_time < persona->agent.proactive_interval) {
                return decision;
            }
            decision.should_speak = true;
            decision.trigger = request.system_event_content.empty()
                ? "[system_event]"
                : request.system_event_content;
            return decision;
        }
        if (persona->agent.proactive_level == ProactiveLevel::Low) {
            return decision;
        }

        if (persona->proactive.enabled) {
            const auto idle_hours = std::chrono::duration<double, std::ratio<3600>>(idle);
            if (idle_hours < persona->proactive.idle_trigger_hours) {
                return decision;
            }
            if (IsSet(persona->last_proactive_time) &&
                now - persona->last_proactive_time < persona->proactive.min_interval) {
                return decision;
            }
            persona->last_proactive_time = now;
            persona->proactive_state = ProactiveState::Triggered;
            decision.state = persona->proactive_state;
            provider = decision_provider_;
            proactive_options = persona->proactive;
            provider_request.classroom_id = classroom_it->first;
            provider_request.persona_id = persona->persona_id;
            provider_request.session_id = persona->session_id;
            provider_request.idle_time = idle;
        } else {
            if (idle < persona->agent.idle_threshold) {
                return decision;
            }
            if (IsSet(persona->last_proactive_time) &&
                now - persona->last_proactive_time < persona->agent.proactive_interval) {
                return decision;
            }
            decision.should_speak = true;
            decision.trigger = ProactiveTrigger(idle);
            return decision;
        }
    }

    if (!provider) {
        std::lock_guard lock(mutex_);
        auto classroom_it = classrooms_.find(request.classroom_id);
        if (classroom_it != classrooms_.end()) {
            if (auto* persona = FindPersonaLocked(classroom_it->second, provider_request.persona_id, provider_request.session_id)) {
                persona->proactive_state = ProactiveState::Normal;
                decision.state = persona->proactive_state;
            }
        }
        return decision;
    }

    auto provider_decision = provider->Decide(provider_request);
    if (!provider_decision.ok()) {
        return provider_decision.status();
    }
    if (provider_decision.value().should_respond &&
        provider_decision.value().confidence >= proactive_options.confidence_threshold) {
        decision.should_speak = true;
        decision.trigger = provider_decision.value().reason.empty()
            ? "[proactive_decision]"
            : "[proactive_decision] " + provider_decision.value().reason;
        return decision;
    }

    std::lock_guard lock(mutex_);
    auto classroom_it = classrooms_.find(request.classroom_id);
    if (classroom_it != classrooms_.end()) {
        if (auto* persona = FindPersonaLocked(classroom_it->second, provider_request.persona_id, provider_request.session_id)) {
            persona->proactive_state = ProactiveState::Normal;
            decision.state = persona->proactive_state;
        }
    }
    return decision;
}

core::Result<ClassroomHealthReport> ClassroomScheduler::Health(std::string_view classroom_id) const {
    std::lock_guard lock(mutex_);
    auto classroom_it = classrooms_.find(std::string(classroom_id));
    if (classroom_it == classrooms_.end()) {
        return core::Status::Error(core::ErrorCode::NotFound, "classroom not found");
    }
    const auto now = std::chrono::steady_clock::now();
    ClassroomHealthReport report;
    report.classroom_id = classroom_it->first;
    for (const auto& [persona_id, persona] : classroom_it->second.personas) {
        ClassroomHealthPersona item;
        item.persona_id = persona_id;
        item.session_id = persona.session_id;
        item.context_ids = SortedSet(persona.context_ids);
        item.context_patterns = persona.context_patterns;
        item.config_source = persona.config_source;
        item.proactive_state = persona.proactive_state;
        item.uptime_seconds = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(now - persona.started_at).count());
        report.personas.push_back(std::move(item));
    }
    std::sort(report.personas.begin(), report.personas.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.persona_id < rhs.persona_id;
    });
    return report;
}

bool ClassroomScheduler::GlobMatch(std::string_view text, std::string_view pattern) {
    std::size_t t = 0;
    std::size_t p = 0;
    std::size_t star = std::numeric_limits<std::size_t>::max();
    std::size_t match = 0;
    while (t < text.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) {
            ++t;
            ++p;
        } else if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            match = t;
        } else if (star != std::numeric_limits<std::size_t>::max()) {
            p = star + 1;
            t = ++match;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') {
        ++p;
    }
    return p == pattern.size();
}

std::string ClassroomScheduler::ProactiveTrigger(std::chrono::seconds idle) {
    const auto minutes = std::chrono::duration_cast<std::chrono::minutes>(idle).count();
    if (minutes > 0) {
        return "conversation idle for " + std::to_string(minutes) + " minutes";
    }
    return "conversation idle for " + std::to_string(idle.count()) + " seconds";
}

core::Result<ClassroomRouteResult> ClassroomScheduler::ResolveLocked(ClassroomState& classroom,
                                                                     std::string_view classroom_id,
                                                                     const ClassroomRouteRequest& request) {
    auto make_result = [&](const ManagedPersona& persona, bool cache_hit, bool default_route) {
        ClassroomRouteResult result;
        result.classroom_id = std::string(classroom_id);
        result.persona_id = persona.persona_id;
        result.session_id = persona.session_id;
        result.context_id = request.context_id;
        result.cache_hit = cache_hit;
        result.default_route = default_route;
        return result;
    };

    if (!request.session_id.empty()) {
        for (const auto& [id, persona] : classroom.personas) {
            if (persona.session_id == request.session_id) {
                return make_result(persona, false, false);
            }
        }
        return core::Status::Error(core::ErrorCode::NotFound, "session not found in classroom");
    }
    if (!request.persona_hint.empty()) {
        auto persona_it = classroom.personas.find(request.persona_hint);
        if (persona_it != classroom.personas.end()) {
            return make_result(persona_it->second, false, false);
        }
    }
    if (!request.context_id.empty()) {
        auto cache_it = classroom.context_cache.find(request.context_id);
        if (cache_it != classroom.context_cache.end()) {
            auto persona_it = classroom.personas.find(cache_it->second);
            if (persona_it != classroom.personas.end()) {
                return make_result(persona_it->second, true, false);
            }
        }
        for (const auto& [id, persona] : classroom.personas) {
            if (persona.context_ids.contains(request.context_id)) {
                classroom.context_cache[request.context_id] = id;
                return make_result(persona, false, false);
            }
        }
        for (const auto& [id, persona] : classroom.personas) {
            if (std::any_of(persona.context_patterns.begin(), persona.context_patterns.end(), [&](const auto& pattern) {
                    return GlobMatch(request.context_id, pattern);
                })) {
                classroom.context_cache[request.context_id] = id;
                return make_result(persona, false, false);
            }
        }
    }

    auto default_id = classroom.default_persona.empty() ? options_.default_persona : classroom.default_persona;
    if (!default_id.empty()) {
        auto persona_it = classroom.personas.find(default_id);
        if (persona_it != classroom.personas.end()) {
            if (!request.context_id.empty()) {
                classroom.context_cache[request.context_id] = default_id;
            }
            return make_result(persona_it->second, false, true);
        }
    }
    return core::Status::Error(core::ErrorCode::NotFound, "no classroom persona matched request");
}

ClassroomScheduler::ManagedPersona* ClassroomScheduler::FindPersonaLocked(ClassroomState& classroom,
                                                                          std::string_view persona_id,
                                                                          std::string_view session_id) {
    if (!persona_id.empty()) {
        auto persona_it = classroom.personas.find(std::string(persona_id));
        if (persona_it != classroom.personas.end()) {
            return &persona_it->second;
        }
    }
    if (!session_id.empty()) {
        for (auto& [id, persona] : classroom.personas) {
            if (persona.session_id == session_id) {
                return &persona;
            }
        }
    }
    return nullptr;
}

std::string ToString(ProactiveLevel level) {
    switch (level) {
    case ProactiveLevel::Off: return "off";
    case ProactiveLevel::Low: return "low";
    case ProactiveLevel::Medium: return "medium";
    }
    return "off";
}

std::string ToString(ProactiveState state) {
    switch (state) {
    case ProactiveState::Normal: return "normal";
    case ProactiveState::Triggered: return "triggered";
    case ProactiveState::WaitingResponse: return "waiting";
    case ProactiveState::Dormant: return "dormant";
    }
    return "normal";
}

ProactiveLevel ProactiveLevelFromString(std::string_view value) {
    if (value == "low") {
        return ProactiveLevel::Low;
    }
    if (value == "medium") {
        return ProactiveLevel::Medium;
    }
    return ProactiveLevel::Off;
}

} // namespace agent::service::gateway
