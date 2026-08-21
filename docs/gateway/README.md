# Gateway Documents

本目录保存浏览器到 Gateway 的公开协议、Gateway/Frontend 生命周期对齐和 Session affinity 调度边界。

## 当前规范与设计

- [Frontend/Backend API Protocol](FRONTEND_BACKEND_API_PROTOCOL.md)
- [Gateway/Frontend Session Alignment](GATEWAY_FRONTEND_SESSION_ALIGNMENT.md)
- [Gateway Session Affinity Scheduler](GATEWAY_SESSION_AFFINITY_SCHEDULER.md)

## 相关边界

- 通用 Session/Skill 状态机由 [Architecture](../architecture/README.md) 维护；
- Gateway 启动配置由 [Runtime](../runtime/README.md) 维护；
- 认证和输入安全由 [Security](../security/README.md) 维护；
- 调度 A/B 和容量数据由 [Performance](../performance/README.md) 维护。

Gateway 协议变更必须同步更新适配器测试、HTTP/WebSocket E2E 和前端调用方，不得只修改示例 payload。
