#pragma once

#include "skill_session_manager.h"
#include "vision_runtime_interfaces.h"

#include <memory>
#include <string>

namespace agent::service::persona {

struct SkillVisionEventSinkOptions {
    std::string skill_id = "vision.observe";
    bool auto_start_missing_session = true;
    bool mark_ready_on_event = true;
    double min_prompt_confidence = 0.0;
};

class SkillVisionEventSink final : public media::IVisionEventSink {
public:
    SkillVisionEventSink(std::shared_ptr<ISkillSessionManager> manager,
                         SkillVisionEventSinkOptions options = {});

    core::Status Publish(media::VisionEvent event) override;

private:
    static std::string ObservationSummary(const media::VisionEvent& event);
    static double ObservationConfidence(const media::VisionEvent& event);

    std::shared_ptr<ISkillSessionManager> manager_;
    SkillVisionEventSinkOptions options_;
};

} // namespace agent::service::persona
