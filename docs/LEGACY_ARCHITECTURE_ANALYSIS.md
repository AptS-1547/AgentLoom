# 旧架构问题分析与重构方向

> 面向团队内部开发人员的历史架构复盘
> 分析日期：2026-05-16
> 依据范围：`D:\Users\21405\source\repos\EducationalAgentProject` 后端、`D:\front_end\Train` 前端关键文档与关键代码

## 1. 结论摘要

旧架构的核心问题不是某一个模块实现错误，而是前端、ADP/LKE 工作流、Python 后端、C++ 推理端和 Nginx 网关之间缺少一个统一的后端协议层。结果是前端 Vite/Node 层在交付期间承担了部分网关、代理、签名、上传转发和报告调度职责；Python 后端虽然提供了 OpenAI-compatible API、安全中间件、人格管线和 C++ ONNX gRPC 推理接入，但没有成为所有外部协议的唯一入口。

这套方案在比赛时间、ADP 平台接入、服务器运维压力和前端快速更新需求下是可以理解的临时架构。它的问题在于不适合作为生产或商业化基础：职责边界模糊、调用链绕行、协议重复、密钥暴露面扩大、部署形态依赖开发服务器、可观测性和故障定位被拆散。

下一阶段应把协议层收束到后端网关：前端只负责静态资源和交互 UI；Nginx/CDN 只负责 TLS、静态资源、粗粒度反代和限流；C++ Gateway 负责 HTTP/WebSocket/SSE 兼容、认证、连接池、背压、上传代理、缓存路由、ADP/第三方平台适配和推理服务调度；Python 保留业务变化快的冷路径能力。

## 2. 背景

旧架构形成于比赛项目和研究原型迁移的交叉阶段。该项目并非从商业化后端架构正向设计，而是从 `MyNeuroLikeSystem` 研究型原型中裁剪、适配和部署出来：原型包含轻量 UI、大量试验性功能以及 QQ 机器人服务，比赛版本需要在短时间内满足网页演示、ADP 平台调用、服务器部署和评审材料要求。

当时的主要约束包括：

- 业务方案和前端页面仍在快速变化，需要前端代码具备快速更新能力。
- ADP/LKE 平台要求通过指定接口形态接入智能体能力。
- 后端智能体、C++ 推理端、前端工程和服务器环境需要同时打通。
- 大量时间消耗在服务器环境、Nginx 转发、Docker 镜像、平台联调和运维上。
- 比赛交付优先级高于长期可维护性，临时折中不可避免。

因此，旧架构应被视为一个在强约束下完成交付的原型架构，而不是生产架构。

## 3. 实际旧架构链路

从两个旧仓库可还原出以下主要链路。

### 3.1 前端 SPA 与 ADP/LKE 直连链路

`D:\front_end\Train\README.md` 将前端定义为 Vue 3 + Vite 单页应用，通过 HTTP / SSE 对接工作流与直连评估接口。实际工作流代码位于 `vendor/front/js/workflow.js`，默认请求腾讯云 LKE `https://wss.lke.cloud.tencent.com/v1/qbot/chat/sse`，通过 `bot_app_key`、`visitor_biz_id`、`custom_variables` 等字段直接与 ADP 工作流交互。

相关调用点：

- `D:\front_end\Train\src\classroom-workflow-inject.js` 将 `VITE_SPECIAL_BOT_APP_KEY`、`VITE_CLASSROOM_BOT_APP_KEY`、`VITE_REPORT_*` 注入到 `window.__WORKFLOW_INJECT__` 等全局对象。
- `D:\front_end\Train\vendor\front\js\workflow.js` 实现专项模拟、课堂模拟和报告相关请求的 fetch/SSE 解析。
- `D:\front_end\Train\src\components\TeachingDocAnalysis.vue` 固定使用官方 qbot SSE 地址处理教学文档分析，将文件 URL 作为工作流输入。

这意味着部分智能体交互并不经过自有后端，而是从浏览器直接进入 ADP/LKE。

### 3.2 Vite/Node 代理链路

`D:\front_end\Train\vite.config.js` 不只是构建配置，还实现了三个后端性质的中间件/代理：

- `POST /api/describe-storage-credential`：读取 `VITE_ADP_SECRET_ID` 和 `VITE_ADP_SECRET_KEY`，在 Node 侧生成 Tencent Cloud TC3-HMAC-SHA256 签名，然后调用 LKE OpenAPI `DescribeStorageCredential`。
- `POST /api/cos-upload`：从请求头读取 `x-upload-url`，读取完整请求 body，再向 COS 预签名地址发起 PUT。
- `/api/report`：通过 Vite proxy 转发到报告/后端 Chat Completions 接口，默认路径为 `/v1/chat/completions`。

`D:\front_end\Train\.env.example` 也明确提示：`VITE_REPORT_HTTP_API_KEY` 会打进前端包、浏览器可见，生产环境建议由自有后端代发；Vite 的 `/api/report` 代理只在 dev / preview 有效，生产需要 Nginx 或后端提供等价网关。

这说明旧前端工具链实际承担了开发期网关职责，并且有一部分安全和上传代理逻辑落在 Vite dev/preview server 中。

### 3.3 旧 Python 后端 OpenAI-compatible 链路

`D:\Users\21405\source\repos\EducationalAgentProject\README.md` 描述后端职责包括 persona、记忆、Prompt、OpenAI 兼容 API、小模型前后处理和多人格调度。核心 API 包括：

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`

`agent/src/adapters/openai_adapter.py` 将 NeuroLike pipeline 包装为 FastAPI 服务，内部运行 BERT 情绪分类、情绪融合、OU 情绪状态机、分级记忆和 LLM 生成，并支持：

- OpenAI Chat Completions 形态响应。
- 文本和图片多模态输入。
- `evaluation` 报告请求。
- `proactive` 主动发言轮询。
- 多人格 `model` 字段路由。
- 文档分析 router。

`agent/src/adapters/security.py` 也实现了认证、限流、IP 封禁和请求大小限制。这说明旧后端并非没有协议和安全能力，但这些能力只覆盖打到 FastAPI 的后端请求，不能统一管理前端直连 ADP、Vite 代理、COS 上传和 Nginx 网关中的全部外部入口。

### 3.4 Python 到 C++ ONNX gRPC 推理链路

`EducationalAgentProject` 支持 `pytorch` 和 `onnx_grpc` 两种小模型后端。README 中的 `onnx_grpc` 链路为：

```text
Python tokenizer -> C++ ONNX gRPC backend -> Python postprocessing
```

这部分已经体现出后端拆分和 C++ 推理加速的方向，但它仍处在 Python agent 的内部依赖关系中。C++ 推理端当时不是统一外部协议入口，而是被 Python 后端调用的推理服务。

### 3.5 Docker Compose 与 Nginx 单域名网关链路

`D:\Users\21405\source\repos\EducationalAgentProject\docker-compose.yml` 同时编排了 `agent` 和 `frontend`：

- `agent` 暴露 `8000`，运行 FastAPI/OpenAI-compatible 服务。
- `frontend` 从 `D:\front_end\Train` 构建，暴露 `5173`，运行 Vite dev server。
- 前端环境变量将 `VITE_REPORT_PROXY_TARGET` 默认指向 `http://agent:8000`，即浏览器请求 `/api/report` 先进入 Vite，再由 Vite 转发到 Python agent。

`D:\Users\21405\source\repos\EducationalAgentProject\deploy\nginx\edu-agent.conf` 更直接说明了旧网关形态：

- `/v1/` 转发到 `127.0.0.1:8000` 后端 agent。
- `/agent/health` 转发到后端健康检查。
- `/api/` 转发到 `127.0.0.1:5173` 前端 Vite 中间件。
- `/` 转发到 Vite dev server，并保留 HMR/WebSocket 支持。

这使得 Nginx 成为单域名入口，但实际业务网关逻辑仍被拆在 Nginx、Vite server、前端 JS、ADP/LKE 和 Python agent 之间。

## 4. 主要问题

### 4.1 前后端职责边界不清

前端工程承担了四类不应长期留在前端工具链中的职责：

- ADP OpenAPI 凭证签名。
- COS 上传代理。
- 报告 API 反向代理。
- 工作流协议字段拼装和 SSE 解析。

这些逻辑和 UI 迭代绑定后，会导致前端构建、运行环境、密钥策略、CORS 策略和后端协议演进互相牵制。Vite 是优秀的前端开发工具，但不应成为生产网关的业务承载层。

### 4.2 后端未成为统一协议入口

Python 后端已经能处理 OpenAI-compatible 请求，也有安全中间件和并发限制。但旧架构中存在多条绕过后端的外部链路：

- 浏览器直接调用 ADP/LKE qbot SSE。
- 浏览器通过 Vite 中间件调用 LKE OpenAPI 获取 COS 凭证。
- 浏览器通过 Vite 中间件转发 COS 上传。
- 浏览器通过 Vite `/api/report` 再转发到后端 agent。
- ADP 或 OpenAI-compatible 客户端直接调用 `/v1/chat/completions`。

这些链路导致认证、限流、审计、日志、trace id、错误包装、超时、重试和资源治理无法在一个入口统一执行。

### 4.3 开发服务器被部署为运行时组件

`D:\front_end\Train\Dockerfile` 设置 `NODE_ENV=development`，并以 `npm run dev -- --host 0.0.0.0 --port 5173` 作为容器入口。旧 Nginx 配置也明确把 `/` 和 `/api/` 转发到 Vite dev server。

这种方式对比赛交付有现实价值：保留热更新能力，减少每次改前端都重新构建和分发静态资源的成本。但生产上存在明显问题：

- Vite dev server 不是面向公网生产流量的稳定网关。
- HMR/WebSocket、源码服务和运行代理混在一起，攻击面扩大。
- 前端源码更新能力与线上运行稳定性冲突。
- 构建产物 `dist/` 的静态部署路径没有成为唯一交付形态。

### 4.4 密钥和认证边界扩大

旧前端 `.env.example` 已经提示 `VITE_REPORT_HTTP_API_KEY` 会进入前端包。只要一个字段通过 `VITE_*` 注入到前端 bundle 或运行时上下文，它就不再是严格意义上的服务端秘密，浏览器、Network 面板、源码映射、DevTools 和抓包工具都可能把它还原出来。

`VITE_ADP_SECRET_ID` 和 `VITE_ADP_SECRET_KEY` 虽然由 Vite Node 中间件在服务端读取，不直接发给浏览器，但它们仍被放在前端服务容器的环境变量中，使前端运行时成为云 API 签名代理。攻击者不需要先打穿后端，只要能复现请求模式、截获代理行为或复用前端可见字段，就可能进一步放大访问面。

生产架构中，密钥边界应尽量集中在后端受控服务中。浏览器侧只应持有短期、低权限、可撤销的 session token，不应掌握或间接驱动云厂商级 OpenAPI secret。

### 4.5 安全层脆弱性与单网关放大效应

旧架构还存在明显的单网关放大效应：Nginx 同时承担外部入口、静态页面转发、`/api/` 代理、`/v1/` 代理和长连接转发。一旦网关层出现解析、缓冲或代理级漏洞，影响范围会直接覆盖整站。

这不是抽象担忧。NGINX 官方在 2026-05-13 发布的 1.30.1 stable 版本公告中，明确列出了 `ngx_http_rewrite_module` 的 buffer overflow 修复，受影响范围为 `0.6.27-1.30.0`；同一公告还修复了 `ngx_http_proxy_module` 的 HTTP/2 request injection 等问题。对旧架构来说，这意味着如果网关版本停留在受影响区间，攻击者只需构造特定请求就可能触发拒绝服务或更严重的异常行为，且影响的是统一入口下的所有业务流量。

换句话说，旧架构不是“有一个 Nginx 在前面”这么简单，而是把整个系统的公网可达性押在单个代理实例和它所承载的多个路由职责上。只要这个点出问题，前端页面、报告接口、文档上传和对话入口都会一起受影响。

### 4.6 调用链冗余和延迟增加

旧架构中的报告请求可能形成以下路径：

```text
Browser
  -> Nginx
  -> Vite dev server /api/report
  -> Python FastAPI /v1/chat/completions
  -> Persona pipeline / LLM client / C++ ONNX gRPC
  -> Python FastAPI
  -> Vite dev server
  -> Browser
```

教学文档分析则可能形成：

```text
Browser
  -> Vite /api/describe-storage-credential
  -> Tencent LKE OpenAPI
  -> Browser
  -> Vite /api/cos-upload or direct COS PUT
  -> COS
  -> Browser
  -> Tencent qbot SSE
  -> Browser
```

这些链路在低并发比赛演示中可以工作，但商业化场景下会增加序列化、跨进程代理、跨公网调用、CORS 处理和错误定位成本。

### 4.7 可观测性和故障定位分散

旧架构的日志来源至少包括：

- 浏览器 console。
- Vite dev server。
- Nginx access/error log。
- Python FastAPI/uvicorn log。
- Python agent 内部 pipeline log。
- C++ ONNX gRPC 推理端 log。
- 腾讯云 LKE/ADP 平台侧不可完全控制的工作流日志。

如果没有统一 trace id 和后端入口，某次请求失败时很难判断问题属于浏览器 CORS、Vite 代理、Nginx rewrite、ADP SSE、Python pipeline、C++ gRPC、LLM 上游还是网络超时。

### 4.8 协议模型重复

旧架构中同时存在多种协议形态：

- 腾讯 qbot SSE 请求和事件流。
- OpenAI Chat Completions JSON。
- OpenAI Chat Completions SSE chunk。
- Vite proxy HTTP 转发。
- COS 预签名 PUT。
- Python 到 C++ gRPC。
- Nginx 对 `/v1/`、`/api/`、`/` 的路径级转发。

协议不是越少越好，但必须有清晰的归属和转换边界。旧架构缺少一个统一 Adapter 层，导致协议转换分散在前端 JS、Vite Node、Python FastAPI 和 Nginx 配置中。

## 5. 为什么当时可接受

旧架构的价值在于快速交付，而不是长期治理。它在当时解决了几个真实问题：

- 前端可以快速更新，不需要每次改页面都完整重建后端镜像。
- 通过 Vite 代理快速绕过了浏览器 CORS 和部分上传限制。
- 通过 Nginx 单域名入口满足了外部访问和 ADP 平台调用要求。
- Python FastAPI 保留了原型中的 persona、记忆、Prompt 和 LLM 业务逻辑，降低重写风险。
- C++ ONNX gRPC 推理链路验证了小模型服务化方向。
- Docker Compose 将 agent 和 frontend 一起拉起，降低现场部署复杂度。

因此，旧方案是比赛约束下的合理折中。问题在于它的折中被部署形态固化后，会阻碍后续商业化架构演进。

## 6. 目标架构方向

目标不是简单地把所有逻辑都塞进 C++，而是建立稳定的协议归属：

```text
Browser / Client
  -> CDN / Nginx
  -> C++ Gateway
       -> Auth / Rate Limit / Trace / Request Size Limit
       -> HTTP / WebSocket / SSE compatibility
       -> Upload Credential Proxy / COS Proxy
       -> ADP / OpenAI-compatible Adapter
       -> Exact / Semantic / RAG / Memory Cache Router
       -> vLLM main path / llama.cpp fallback / C++ ONNX gRPC
       -> Python services for memory, RAG, evaluation, document analysis
```

职责建议：

| 层 | 应承担职责 | 不应承担职责 |
|----|------------|--------------|
| 前端 Vue/Vite | UI、交互状态、静态资源构建、浏览器侧展示 | 云 API 签名、生产代理、长期连接治理、服务间调度 |
| Nginx/CDN | TLS、静态资源、粗粒度反代、基础限流、压缩 | 业务路由、协议转换、智能体调度 |
| C++ Gateway | HTTP/WebSocket/SSE 接入、认证、连接池、背压、上传代理、缓存路由、推理路由、统一错误包装 | 复杂业务策略训练、重型 RAG 构建、频繁变动的教学业务逻辑 |
| Python 服务 | 记忆压缩、RAG、文档分析、评估、快速变化的业务编排 | 公网协议入口、长连接管理、高频热路径扫描 |
| C++ 推理端 | ONNX/BERT/VLM/向量核/本地缓存/资源治理 | 前端协议兼容和业务页面适配 |

## 7. 迁移建议

### 7.1 第一阶段：整理旧接口调用

先整理旧系统所有外部入口和调用方：

- `/v1/chat/completions`
- `/v1/models`
- `/health` 或 `/agent/health`
- `/api/report`
- `/api/describe-storage-credential`
- `/api/cos-upload`
- Tencent `qbot/chat/sse`
- COS upload URL
- 前端 `VITE_*` 环境变量

输出一份 endpoint contract，明确每个接口的请求字段、认证方式、响应格式、错误格式、调用方和替代路径。

### 7.2 第二阶段：替换 Vite 运行时代理
### 注意：考虑到ADP仅仅是比赛要求，且具有其不透明、调试困难且需要增加额外成本的特点，后续ADP依赖将会被完全移除

将以下逻辑从 `vite.config.js` 迁移到后端 Gateway 或独立后端 API：

- COS 上传代理。
- `/api/report` 代理。
- 生产环境 CORS 和同源路径管理。

前端生产环境只发布 `npm run build` 的 `dist/`。Vite dev server 只保留为本地开发工具。

### 7.3 第三阶段：统一智能体入口

将浏览器直连 ADP/LKE 的链路逐步改为：

```text
Browser -> Gateway -> Internal Agent
```

Gateway 负责：

- 将浏览器 session token 转换为后端内部凭据。
- 隐藏 `bot_app_key` 和云平台 secret。
- 统一 SSE 解析、错误包装和超时控制。
- 记录 trace id 和调用链耗时。
- 后续把 ADP 工作流替换为自有 RAG/LLM/agent 编排。

### 7.4 第四阶段：后端协议层接管连接治理

基于当前 `AgentBackendPredict` 已有的 HTTP/WebSocket runtime、连接池、线程池、内存池、统一 RAII 和 request interface，逐步实现：

- HTTP handler 异步回调。
- WebSocket 会话池。
- 后端超时关闭和拒绝访问。
- 读写背压。
- 大帧和分片处理。
- 严格请求体大小限制。
- 统一错误和日志格式。

这一步完成后，Python FastAPI 可以退到内部业务服务，不再直接承担公网协议入口。

### 7.5 第五阶段：缓存和推理路由接入

在 Gateway 中加入对话热路径路由：

```text
Normalize / Intent / Context Risk
  -> Exact Cache
  -> Curated Semantic Cache
  -> Knowledge RAG Cache
  -> Memory Context Layer
  -> vLLM
  -> llama.cpp fallback
```

这样可以从架构层避免旧系统中所有请求都倾向进入在线 LLM 或外部工作流平台的问题。

## 8. 风险控制

迁移过程中需要避免一次性推倒重来。建议采用可回滚的双轨策略：

- 保留旧 `/v1/chat/completions` FastAPI 路径，Gateway 初期只做透明反代和 trace 注入。
- 前端先把 `/api/report`、上传代理切到 Gateway，专项/课堂 qbot SSE 后迁移。
- 每迁移一个入口，保留同名兼容路径和灰度开关。
- 所有新 Gateway 路径必须具备单元测试和集成测试，特别是请求大小限制、SSE 断流、超时、认证失败和并发 Submit。
- 不在前端继续新增任何生产代理逻辑。

## 9. 对当前 AgentBackendPredict 以及整个新生产级MVP项目的价值和意义

当前 `AgentBackendPredict` 已经从单一推理端向基础设施层演进，具备继续承接旧架构协议层的基础：

- `src/core`：内存池、RAII、线程池、生产消费队列等基础设施。
- `src/net`：HTTP/WebSocket runtime、连接池和 request interface。
- `src/config`：配置解析和 CLI fallback。
- `src/vector` / `src/cache`：向量缓存和后续 Redis/Faiss/SQLite 接入基础。
- `src/server`：gRPC 内部交换协议和 server common。

因此，旧架构重构不应只是“替换前端代理”，而应把这些基础设施变成正式 Gateway 服务。它负责把历史上分散在 Vite、Nginx、FastAPI 和浏览器 JS 中的协议职责收束起来，并为后续语义缓存、RAG、vLLM 主路径、llama.cpp 备用路径提供统一入口。

## 10. 证据来源

本分析读取了以下关键文件：

| 仓库 | 文件 | 观察点 |
|------|------|--------|
| `D:\front_end\Train` | `README.md` | 前端为 Vue 3 + Vite SPA，通过 HTTP/SSE 对接工作流和评估接口 |
| `D:\front_end\Train` | `Dockerfile` | 生产容器入口实际为 `npm run dev -- --host 0.0.0.0 --port 5173` |
| `D:\front_end\Train` | `vite.config.js` | Vite 中实现 ADP 凭证签名、COS 上传代理、报告接口代理 |
| `D:\front_end\Train` | `.env.example` | 明确提示部分 API key 会进入前端包，生产建议后端代理 |
| `D:\front_end\Train` | `docs\SimuTeach-实际系统架构.md` | 明确 Vite `/api/*` 仅 dev/preview 有效，生产需要等价网关或后端 |
| `D:\front_end\Train` | `docs\第7章-技术实现材料说明.md` | 描述 qbot SSE、DescribeStorageCredential、COS proxy、报告接口 |
| `D:\front_end\Train` | `src\classroom-workflow-inject.js` | 将 `VITE_*` 注入到工作流全局配置 |
| `D:\front_end\Train` | `vendor\front\js\workflow.js` | 前端直接构造 qbot SSE 请求并解析事件流 |
| `D:\front_end\Train` | `src\components\TeachingDocAnalysis.vue` | 文档分析固定使用 qbot SSE，上传链路依赖凭证和 COS |
| `D:\Users\21405\source\repos\EducationalAgentProject` | `README.md` | Python 后端职责、OpenAI-compatible API、C++ ONNX gRPC 推理链路 |
| `D:\Users\21405\source\repos\EducationalAgentProject` | `docker-compose.yml` | agent 和 frontend 并排编排，前端 dev server 作为服务运行 |
| `D:\Users\21405\source\repos\EducationalAgentProject` | `agent\src\adapters\run_api_server.py` | Uvicorn/FastAPI API server 启动入口 |
| `D:\Users\21405\source\repos\EducationalAgentProject` | `agent\src\adapters\openai_adapter.py` | OpenAI-compatible API、人格式路由、evaluation/proactive/document router |
| `D:\Users\21405\source\repos\EducationalAgentProject` | `agent\src\adapters\security.py` | 后端认证、限流、IP 封禁、请求体大小限制 |
| `D:\Users\21405\source\repos\EducationalAgentProject` | `agent\docs\API.md` | 后端 OpenAI-compatible API 形态和 ADP 自定义模型接入方式 |
| `D:\Users\21405\source\repos\EducationalAgentProject` | `deploy\nginx\edu-agent.conf` | 单域名网关将 `/v1/` 打到后端，`/api/` 和 `/` 打到 Vite dev server |

By Orange and GPT5.5
已经过审查
2026.5.16