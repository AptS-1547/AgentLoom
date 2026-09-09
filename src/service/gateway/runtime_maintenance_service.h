#pragma once

#include "document_analysis_service.h"
#include "gateway_maintenance.h"
#include "long_term_memory_compressor.h"
#include "skill_session_manager.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace media {
class WebRtcSessionRegistry;
}

namespace ipc::media {
class IInferenceFrameIpcLeaseCoordinator;
}

namespace agent::service::gateway {

class DocumentRetentionMaintenanceTask final : public IRuntimeMaintenanceTask {
public:
    DocumentRetentionMaintenanceTask(std::shared_ptr<document::DocumentAnalysisService> documents,
                                     std::chrono::milliseconds interval);

    std::string_view Name() const noexcept override;
    std::chrono::milliseconds Interval() const noexcept override;
    core::Status Tick(std::stop_token stop_token) override;

private:
    std::shared_ptr<document::DocumentAnalysisService> documents_;
    std::chrono::milliseconds interval_;
};

class WebRtcSessionMaintenanceTask final : public IRuntimeMaintenanceTask {
public:
    WebRtcSessionMaintenanceTask(std::shared_ptr<media::WebRtcSessionRegistry> registry,
                                 std::chrono::milliseconds interval,
                                 core::LoggerAdapter logger = core::LoggerAdapter::ForModule("gateway"));

    std::string_view Name() const noexcept override;
    std::chrono::milliseconds Interval() const noexcept override;
    core::Status Tick(std::stop_token stop_token) override;

private:
    std::shared_ptr<media::WebRtcSessionRegistry> registry_;
    std::chrono::milliseconds interval_;
    core::LoggerAdapter logger_;
};

class SkillSessionMaintenanceTask final : public IRuntimeMaintenanceTask {
public:
    SkillSessionMaintenanceTask(std::shared_ptr<persona::ISkillSessionManager> manager,
                                std::chrono::milliseconds interval,
                                core::LoggerAdapter logger = core::LoggerAdapter::ForModule("gateway"));

    std::string_view Name() const noexcept override;
    std::chrono::milliseconds Interval() const noexcept override;
    core::Status Tick(std::stop_token stop_token) override;

private:
    std::shared_ptr<persona::ISkillSessionManager> manager_;
    std::chrono::milliseconds interval_;
    core::LoggerAdapter logger_;
};

class InferenceFrameIpcPeerMaintenanceTask final : public IRuntimeMaintenanceTask {
public:
    InferenceFrameIpcPeerMaintenanceTask(
        std::shared_ptr<ipc::media::IInferenceFrameIpcLeaseCoordinator> coordinator,
        std::chrono::milliseconds interval,
        bool auto_recover = true);

    std::string_view Name() const noexcept override;
    std::chrono::milliseconds Interval() const noexcept override;
    core::Status Tick(std::stop_token stop_token) override;

private:
    std::shared_ptr<ipc::media::IInferenceFrameIpcLeaseCoordinator> coordinator_;
    std::chrono::milliseconds interval_;
    bool auto_recover_ = true;
};

struct L3MemoryFlushMaintenanceOptions {
    std::chrono::milliseconds interval{std::chrono::minutes(5)};
    int flush_hour = 3;
    int flush_minute = 0;
    int flush_date_offset_days = 0;
    bool defer_when_sessions_active = true;
    std::vector<memory::MemoryOwner> owners;
};

class L3MemoryFlushMaintenanceTask final : public IRuntimeMaintenanceTask {
public:
    L3MemoryFlushMaintenanceTask(std::shared_ptr<memory::LongTermMemoryCompressor> compressor,
                                 const persona::ISessionManager& sessions,
                                 L3MemoryFlushMaintenanceOptions options,
                                 core::LoggerAdapter logger = core::LoggerAdapter::ForModule("gateway"));

    std::string_view Name() const noexcept override;
    std::chrono::milliseconds Interval() const noexcept override;
    core::Status Tick(std::stop_token stop_token) override;

private:
    std::string TargetDate() const;
    bool IsDue() const;

    std::shared_ptr<memory::LongTermMemoryCompressor> compressor_;
    const persona::ISessionManager& sessions_;
    L3MemoryFlushMaintenanceOptions options_;
    core::LoggerAdapter logger_;
    std::unordered_set<std::string> flushed_dates_;
    mutable std::mutex mutex_;
};

} // namespace agent::service::gateway
