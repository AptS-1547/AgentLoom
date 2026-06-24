#include "skill_vision_event_sink.h"

#include "third_party/nlohmann/json.hpp"

#include <algorithm>
#include <utility>

namespace agent::service::persona {
namespace {

using Json = nlohmann::json;

} // namespace

SkillVisionEventSink::SkillVisionEventSink(std::shared_ptr<ISkillSessionManager> manager,
                                           SkillVisionEventSinkOptions options)
    : manager_(std::move(manager)), options_(std::move(options)) {}

core::Status SkillVisionEventSink::Publish(media::VisionEvent event) {
    if (!manager_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "skill session manager is not configured");
    }
    if (event.session_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "vision event session_id is empty");
    }

    auto current = manager_->Get(event.session_id, options_.skill_id);
    if (!current.ok()) {
        return current.status();
    }
    if (!current.value().has_value()) {
        if (!options_.auto_start_missing_session) {
            return core::Status::Ok();
        }
        SkillSessionStartRequest start;
        start.skill_id = options_.skill_id;
        start.session_id = event.session_id;
        start.trace_id = event.trace_id;
        start.source = "vision_event";
        start.reason = "vision event arrived";
        auto started = manager_->Start(start);
        if (!started.ok()) {
            return started.status();
        }
    }

    if (options_.mark_ready_on_event) {
        auto ready = manager_->MarkReady(event.session_id, options_.skill_id, "vision event stream ready", event.trace_id);
        if (!ready.ok() && ready.code() != core::ErrorCode::FailedPrecondition) {
            return ready;
        }
    }

    const double confidence = ObservationConfidence(event);
    Json metadata{
        {"event_id", event.event_id},
        {"peak_frame_id", event.peak_frame_id},
        {"representative_frame_id", event.representative_frame_id},
        {"peak_score", event.peak_score},
        {"duplicate", event.duplicate},
        {"rate_limited", event.rate_limited},
    };
    if (event.analysis) {
        metadata["facts"] = event.analysis->facts;
        metadata["weak_interpretations"] = event.analysis->weak_interpretations;
        metadata["scene"] = event.analysis->scene;
    }

    SkillObservation observation;
    observation.skill_id = options_.skill_id;
    observation.session_id = event.session_id;
    observation.trace_id = event.trace_id;
    observation.summary = ObservationSummary(event);
    observation.confidence = confidence;
    observation.stale = event.rate_limited || event.duplicate;
    observation.should_inject_prompt = confidence >= options_.min_prompt_confidence &&
                                       !observation.stale &&
                                       !observation.summary.empty();
    observation.source = "vision_event";
    observation.metadata_json = metadata.dump();
    return manager_->RecordObservation(observation);
}

std::string SkillVisionEventSink::ObservationSummary(const media::VisionEvent& event) {
    return event.SummaryText();
}

double SkillVisionEventSink::ObservationConfidence(const media::VisionEvent& event) {
    if (event.analysis) {
        return event.analysis->confidence;
    }
    return std::max(0.0, std::min(1.0, event.peak_score));
}

} // namespace agent::service::persona
