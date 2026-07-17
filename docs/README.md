# AgentLoom Documentation

本文档目录同时保存当前规范、实现说明、实验记录和早期架构资料。阅读时应先确认文档状态；历史文档中的旧项目名、路径和阶段判断用于保留开发上下文，不代表当前公开 API 或部署方式。

## 文档状态

| 状态 | 含义 |
|------|------|
| 当前规范 | 与当前源码边界一致，修改相关实现时应同步更新 |
| 当前设计 | 已确定方向，部分生产组装或验收仍在进行 |
| 实现参考 | 描述一个具体模块、实验或部署路径，使用前应结合源码核对 |
| 历史记录 | 保存早期方案、迁移过程或商业化讨论，不作为当前实现承诺 |

## 新读者入口

建议按以下顺序了解项目：

1. [项目 README](../README.md)：定位、构建目标、模块概览和快速开始。
2. [架构总览](ARCHITECTURE_VISUAL.md)：当前模块、进程边界和主要数据流。
3. [配置系统](CONFIG_SYSTEM.md)：Server 配置和 CLI 覆盖方式。
4. [部署指南](DEPLOYMENT.md)：推理 Server 的构建、启动和运行时依赖。
5. [Frontend/Backend API](FRONTEND_BACKEND_API_PROTOCOL.md)：Gateway HTTP/WebSocket 协议。
6. [Skill Session Protocol](SKILL_SESSION_PROTOCOL.md)：有限工具会话的通用状态机和资源边界。
7. [安全工程准则](SECURITY_ENGINEERING_STANDARD.md)：认证、密钥、日志、输入和依赖安全要求。
8. [扩展 AgentLoom](EXTENDING_AGENTLOOM.md)：下游库依赖、Server 边界和领域 provider 注入。

## 当前规范与设计

| 文档 | 状态 | 内容 |
|------|------|------|
| [CONFIG_SYSTEM.md](CONFIG_SYSTEM.md) | 当前规范 | JSON section、CLI override 和配置扩展方式 |
| [FRONTEND_BACKEND_API_PROTOCOL.md](FRONTEND_BACKEND_API_PROTOCOL.md) | 当前规范 | HTTP、WebSocket、认证、文档和 Skill API |
| [GATEWAY_FRONTEND_SESSION_ALIGNMENT.md](GATEWAY_FRONTEND_SESSION_ALIGNMENT.md) | 当前设计 | Gateway、前端和 session 生命周期对齐 |
| [SKILL_SESSION_PROTOCOL.md](SKILL_SESSION_PROTOCOL.md) | 当前规范 | 通用 Skill Session 生命周期与接口约束 |
| [SKILL_MEDIA_DRAIN_SPOOL_DESIGN.md](SKILL_MEDIA_DRAIN_SPOOL_DESIGN.md) | 当前设计 | 有限媒体输入的 Seal、Drain、mapped spool 和关闭语义 |
| [MULTIMODAL_PERCEPTION_LAYERS.md](MULTIMODAL_PERCEPTION_LAYERS.md) | 当前设计 | ViT/VLM 分层、视觉事件可信度和隐私边界 |
| [STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md) | 当前设计 | WebRTC、GStreamer、抽帧和视觉事件链路 |
| [CONVERSATION_CACHE_AND_INFERENCE_STRATEGY.md](CONVERSATION_CACHE_AND_INFERENCE_STRATEGY.md) | 当前设计 | 对话缓存、推理路由和降级策略 |
| [SECURITY_ENGINEERING_STANDARD.md](SECURITY_ENGINEERING_STANDARD.md) | 当前规范 | 服务端安全工程约束 |
| [EXTENDING_AGENTLOOM.md](EXTENDING_AGENTLOOM.md) | 当前规范 | 下游依赖、CMake alias 和业务扩展接口 |

## 模块与运行参考

| 文档 | 状态 | 内容 |
|------|------|------|
| [DEPLOYMENT.md](DEPLOYMENT.md) | 实现参考 | Windows/Linux 推理服务部署 |
| [GSTREAMER_WINDOWS_SETUP.md](GSTREAMER_WINDOWS_SETUP.md) | 实现参考 | Windows GStreamer 安装与插件检查 |
| [NET_API_NOTES.md](NET_API_NOTES.md) | 实现参考 | Net 层接口和设计说明 |
| [REDIS_COMPATIBILITY.md](REDIS_COMPATIBILITY.md) | 实现参考 | Redis/Redis++ 兼容性注意事项 |
| [PHASE2_EMBEDDING_MODEL.md](PHASE2_EMBEDDING_MODEL.md) | 实现参考 | ONNX text embedding 导出与测试 |
| [E2E_TEST_AND_EMOTION_PIPELINE.md](E2E_TEST_AND_EMOTION_PIPELINE.md) | 实现参考 | VLM cache E2E 和情绪链路讨论 |
| [EMOTION_FUSION_GATE_EXPERIMENT.md](EMOTION_FUSION_GATE_EXPERIMENT.md) | 实验记录 | MAP 标定、LLM gate 和 domain shift 结论 |
| [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) | 实验记录 | 特定版本和硬件条件下的性能数据 |
| [VISION_SYSTEM_DESIGN.md](VISION_SYSTEM_DESIGN.md) | 实现参考 | 视觉系统总体设计 |
| [VISION_MIGRATION.md](VISION_MIGRATION.md) | 实现参考 | 视觉链路本地化迁移思路 |
| [VISION_PIPELINE_ANALYSIS.md](VISION_PIPELINE_ANALYSIS.md) | 实验记录 | 早期视觉链路和项目边界分析 |
| [VISUAL_TOOL_SESSION_PROTOCOL.md](VISUAL_TOOL_SESSION_PROTOCOL.md) | 实现参考 | 视觉工具会话协议的早期专用版本 |

## 历史与迁移记录

以下文档保留设计演进价值，但不应直接用于判断当前 API、构建目标或完成度：

- [NEXT_RUNTIME_ROADMAP.md](NEXT_RUNTIME_ROADMAP.md)：旧 Runtime 路线图。
- [INFRASTRUCTURE_PLAN.md](INFRASTRUCTURE_PLAN.md)：基础设施早期开发计划。
- [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) / [ARCHITECTURE_VISUAL.md](ARCHITECTURE_VISUAL.md)：早期教育智能体商业架构图。
- [COMMERCIAL_ARCHITECTURE.md](COMMERCIAL_ARCHITECTURE.md) / [SUMMARY.md](SUMMARY.md)：早期商业化架构讨论和总结。
- [LEGACY_ARCHITECTURE_ANALYSIS.md](LEGACY_ARCHITECTURE_ANALYSIS.md)：旧 Python/前端架构问题分析。
- [FRONTEND_REFACTOR_E2E_PLAN.md](FRONTEND_REFACTOR_E2E_PLAN.md)：前端迁移阶段计划。
- [TEAM_IMPLEMENTATION_PLAN.md](TEAM_IMPLEMENTATION_PLAN.md)：早期团队实施建议。
- [CACHE_AND_OPTIMIZATION.md](CACHE_AND_OPTIMIZATION.md)：缓存和视觉对话优化草案。
- [REDIS_PLUSPLUS_MIGRATION.md](REDIS_PLUSPLUS_MIGRATION.md) / [REDIS_PLUSPLUS_MIGRATION_COMPLETE.md](REDIS_PLUSPLUS_MIGRATION_COMPLETE.md)：Redis++ 迁移过程记录。

## 维护规则

- 当前规范应在相关接口、配置字段或生命周期语义变化时同步更新。
- 实验数据必须写明日期、输入、硬件、构建配置和适用边界。
- 历史文档不做无意义的全量改名；如结论已失效，在文首增加状态说明。
- 文档中的命令、路径和示例配置默认使用 UTF-8，并使用占位路径或 `.example` 文件，不提交本机配置。
- 新增文档时应同时更新本索引，并明确其状态。
