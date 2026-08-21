# AgentLoom 服务端安全工程准则

本文档定义 AgentLoom 核心库、基础 Server、测试工具和下游扩展模块的最低安全要求。它是工程约束，不替代部署组织自己的威胁建模、合规审查和应急流程。

## 1. 信任边界

- HTTP、WebSocket、WebRTC signaling、gRPC、Redis、SQLite 文件、模型文件、文档上传和共享内存元数据均视为不可信输入。
- Gateway、BERT/VLM inference server 和外部 LLM backend 保持独立故障域；跨进程失败统一映射为 `core::Status` 和协议层状态。
- 共享内存只传输固定布局元数据和帧 payload，不保存进程地址、裸指针或 STL 容器。
- 下游领域模块通过接口注入，不应要求核心库读取专有密钥、指标配置或业务数据库。

## 2. 密钥和凭据

- API key、Redis 密码、TLS 私钥和认证 token 不得提交到仓库、example 配置、日志、dump 或测试报告。
- example 配置只声明环境变量名、key file 路径或空占位值。
- 服务端优先从环境变量或受控 key file 读取凭据；生产环境不得通过浏览器可见的前端变量分发长期云端密钥。
- 日志不得打印完整 Authorization header、cookie、JWT、API key、Redis 密码或 ICE credential。
- 凭据加载失败使用 `core::Status` 返回，并在日志中记录配置来源和 trace，不记录凭据内容。

## 3. 输入验证

- 所有协议入口必须限制消息体大小、字符串长度、数组数量、模型 token 数和并发数量。
- 文件上传同时校验扩展名、大小、ZIP/OOXML 结构、路径归属和最终存储位置。
- 路径规范化后必须验证仍位于允许的根目录内；不得依赖字符串前缀判断目录归属。
- JSON、protobuf、Redis payload 和共享内存 header 在访问字段前验证 schema/version、范围和完整长度。
- WebSocket、WebRTC close/resume、ICE 和 session 操作必须具备幂等或明确的重复请求行为。

## 4. 认证和会话

- 生产 API 默认要求认证；开发注册和路径测试端点必须通过显式配置启用。
- 资源访问必须校验认证用户与 session/document/skill 的所有权，不能只校验资源 ID 是否存在。
- Cookie 使用 `HttpOnly`、合适的 `SameSite` 和生产环境 `Secure` 属性。
- token 必须有过期时间、唯一标识和撤销路径；认证失败不返回内部解析细节。
- Redis/SQLite session store 的失败不得自动退化为绕过认证。

## 5. 错误、异常和日志

- C API 错误码、库异常和业务错误统一包装为 `core::Status` / `core::Result<T>`。
- 业务失败在返回 Status 的同时记录模块、操作、trace ID、session ID 和安全的错误摘要。
- 协议响应不暴露文件系统绝对路径、堆栈、SQL、密钥或模型内部缓冲信息。
- 未知异常在进程边界统一清洗；不得让异常穿过 C ABI、线程入口或 gRPC handler。
- 对认证失败、限流、输入拒绝、模型 OOM、IPC epoch 错误和异常关闭维护可观测计数。

## 6. 资源与并发安全

- HANDLE、文件、SQLite/Redis 连接、ONNX/llama context、GStreamer object 和共享内存映射使用 RAII。
- 队列、spool、连接池、session registry、模型 runner pool 和上传缓存必须有容量上限。
- GStreamer callback、WebSocket IO 线程和 gRPC handler 不执行无界阻塞模型推理。
- Skill 媒体输入在 admission 前检查配额；“不静默丢失”只适用于已成功接纳的数据。
- 进程停止和异常回滚必须覆盖线程 join、claim acknowledge、临时文件清理和数据库事务释放。

## 7. 依赖、模型和数据

- 发布前核对第三方源码、二进制、模型和数据集的许可证与再分发条件。
- 不可信模型、tokenizer、插件和动态库不得在生产主机上自动下载并立即加载。
- 依赖版本、哈希和构建工具链应可追溯；Windows 的 CMake generator、MSVC toolset 和 vcpkg ABI 必须一致。
- 测试图片、对话样本和 dump 在提交前检查个人信息、身份信息和版权。
- VLM 单帧输出不作为身份、性别、情绪或长期事实的可信来源。

## 8. 测试要求

安全相关模块至少覆盖：

- 输入边界和畸形 payload；
- 未认证、越权和过期 token；
- 路径穿越、损坏 ZIP/OOXML 和超限上传；
- 日志不包含凭据；
- Redis/SQLite/IPC/模型失败路径；
- 重复 close、断线恢复、进程退出和资源释放；
- 跨平台条件编译和 UTF-8 路径。

发现安全问题时，优先通过 GitHub Security Advisory 私下报告。不要在公开 Issue 中附带真实凭据、可直接利用的生产地址或包含个人数据的 dump。
