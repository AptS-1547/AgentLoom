#pragma once

#include "result.h"

#include <nlohmann/json.hpp>

#include <string>

namespace agent::service::gateway {

struct ReportEvaluationRequest {
    std::string trace_id;
    std::string session_id;
    std::string user_uuid;
};

/// 下游领域报告评估扩展；实现自行持有数据库、缓存和指标配置。
class IReportEvaluator {
public:
    virtual ~IReportEvaluator() = default;

    /// 生成领域评估结果。
    /// @param request 当前报告的 trace、session 和用户标识。
    /// @return 成功时返回可序列化 JSON；失败不会使 Gateway 基础 metrics 报告失效。
    virtual core::Result<nlohmann::json> Evaluate(
        const ReportEvaluationRequest& request) = 0;
};

} // namespace agent::service::gateway
