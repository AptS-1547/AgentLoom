# Skill Tool-Calling 运行时（2026-09）

> 状态：当前实现 + 测试结论
>
> 范围：`src/skill/` 工具调用协议层、L4 工具记忆召回、Persona 异步工具调用收口
>
> 面向：下游 Application 策略物化层的复用（工具工作流 + grounding + 结构化结果回填）

## 1. 实现现状

`src/skill/` 是一层独立的 **LLM 工具调用协议运行时**，把「工具声明 → 调用解析 → 执行 → 结果回填」这一 OpenAI-compatible function-calling 协议独立成模块，与既有的 Skill Session 生命周期（`skill_session_manager`，面向媒体/视觉长任务）解耦。

| 文件 | 职责 |
|---|---|
| `skill_manifest.h` | `SkillManifest` / `SkillExecutorSpec` / `ExecutionMode` 与 `ValidateManifest()` |
| `skill_registry.h/.cpp` | `ISkillRegistry` + `InMemorySkillRegistry`，按 `skill_id@version` 版本化存储，`tool_name` 全局唯一 |
| `skill_prompt_compiler.h/.cpp` | `ParseToolCalls()`、`MakeToolResultMessage()`、`ISkillPromptCompiler` |
| `skill_executor.h/.cpp` | `ISkillExecutor` 工厂、`SkillInvocationService`、`SkillToolCallCoordinator`（同步/异步编排） |
| `skill_manifest_json.h` | manifest 字段映射（内联，供 registry 与配置段共用，避免重复） |
| `config/sections/skill_registry_config_section.cpp` | `skills` 配置段：内联 manifest 或目录加载（UTF-8 BOM 兼容、文件名 regex、相对路径解析） |

关键抽象均以 `I` 前缀抽象类呈现（`ISkillRegistry` / `ISkillExecutor` / `ISkillExecutorFactory` / `ISkillInvocationService` / `ISkillToolCallCoordinator` / `ISkillPromptCompiler`），当前生产实现为 `InMemory*` 系列，持久化/插件化（WASM/动态加载）作为后续按需接入点。

**L4 工具记忆**（`tool_memory_provider`）负责在 LLM 上下文里注入「该用哪些工具、怎么用」：向量召回（语义）+ 关键词正则（确定性）双通道，输出可注入 prompt 的工具指令块与工具定义。

## 2. 工作原理

### 2.1 工具调用协议（两轮）

```text
LLM 首轮（tool_round=0）
  -> 返回 tool_calls
  -> coordinator 解析 + 校验（必填参数、schema）
  -> 执行器 Start，经 Skill Session 生命周期执行
  -> 回调产生 SkillResult
  -> MakeToolResultMessage 编码为 role=tool 消息
  -> 回填 assistant(tool_calls) + tool 消息后，重投 LLM follow-up（tool_round=1）
```

要点：`ParseToolCalls` 校验必填参数并把 `call.tool_id` 归一化为真实 `skill_id`+`version`；`MakeToolResultMessage` 对成功回填 `result_json`、失败回填结构化 `{"error":{code,message}}`；prompt 编译产出 `<skill_tool_protocol>` 约束块 + 每工具 `<skill_tool>` 描述，强调「工具返回前不得声称外部结果」。

### 2.2 工具召回（关键词正则 + 向量）

`VectorToolMemoryProvider::Query` 组合两条通道：

- **向量通道**：query 编码 → 检索同 partition 的 `capability` 条目 → 按 `top_k`/`min_score` 取前若干，命中项从 `extra_metadata_json` 读 `tool_id`/`instruction`/`schema`。
- **关键词正则通道**：每个工具配置 `keywords`（正向触发）与 `negative_keywords`（否定排除），由 provider 按固定格式（逐词转义 + 竖线交替，对齐下游 `behavior_rule_config` 的 `QuoteMeta` 语义）编译成正则；命中正向词且未命中否定词时，确定性返回该工具。

两通道结果按 `tool_id` 去重、按 `priority`→`score` 排序后截断到 `top_k`，形成注入 prompt 的工具指令块与可调用工具定义。

### 2.3 异步收口

`persona_runtime` 在首轮 LLM 返回非空 `tool_calls` 且 `tool_round==0` 时，经 `ISkillToolCallCoordinator::ExecuteAsync` 异步执行工具并重投 follow-up，**释放 worker**；回调仅做状态更新与 continuation 投递。`SkillInvocationService` 把执行结果路由回 Skill Session 生命周期（终态 `Closed`/`Failed`），并维护 `active_executions_` 以支持取消。

## 3. 测试结果

### 3.1 单元测试

`src/skill/` 相关单测 28 例全通过，覆盖 manifest 校验/版本化注册去重、调用解析与必填参数校验、执行器工厂按 type/reference 解析、调用服务生命周期路由、协调器同步/异步/超时取消，以及 `PersonaRuntimeTest` 的异步工具调用收口。

### 3.2 真实模型 E2E（deepseek 系列）

| 报告 | 用例 | 结论 |
|---|---|---|
| 基础工具调用 | 4/4 | 显式/隐式请求、无工具请求、缺参压力全通过 |
| L4 工具记忆注入 + follow-up | 4/4 | L4 命中注入、两轮 follow-up 收口 |
| 复杂/对抗性 | 11/11 | 长上下文、多任务、明确拒绝、prompt 注入、歧义指代、结构化约束全通过 |

关键结论：结构化 `input_schema` + grounding 提示能稳定压制「无中生有」；prompt 注入用例中模型按协议先调用工具等待真实结果而非直接声称。

### 3.3 多工具召回判别（4 工具 14 用例）

在 `vision.observe` / `document.analyze` / `kb.query` / `profile.read` 四工具判别矩阵（正样本 + 近邻负样本 + 无关负样本）上：

| 召回策略 | 召回通过 | top-1 准确率 | 误召回 |
|---|---|---|---|
| 纯向量 | 12/14 | 9/11 | 0 |
| + 关键词正则（正向） | 13/14 | 10/11 | 1 |
| + 否定排除词 | **14/14** | **11/11** | **0** |

结论：**关键词正则（正+负）+ 向量召回**的组合在多工具场景达到 100% 召回、0 误召回。向量召回解决语义泛化，关键词正则解决近义碰撞与否定排除，二者互补。

## 4. 优势与缺陷

### 4.1 优势

- **协议闭环**：工具声明/解析/执行/回填/follow-up 全链路成型，有单测 + 真实模型 E2E 背书，可直接作为下游策略物化层的工具工作流基础。
- **grounding 有效**：`<skill_tool_protocol>` 约束 + 结构化 schema 能稳定抑制「编造工具结果」。
- **可扩展**：所有核心角色以 `I+` 抽象类呈现，manifest 即扩展契约；JSON 配置声明工具，C++ 只保留执行逻辑。
- **召回互补**：关键词正则（确定性）与向量召回（语义）互补，解决单一通道的近义碰撞/否定盲点。
- **异步不阻塞**：工具执行异步化，worker 全程释放。

### 4.2 缺陷与待办

- **真实 executor 未接入**：`ISkillExecutor` 目前只有测试实现，业务工具（查 RAG/图谱/画像）需下游实现并注册。
- **provenance 链路未落地**：`SkillResult.provenance_json` 字段已声明但未在 `MakeToolResultMessage` 中并入输出，`kb_version`/`evidence_state` 尚需接线。
- **manifest 并发/确认语义未执行**：`execution_mode`、`max_parallel_per_session`、`requires_confirmation` 目前仅声明与校验，运行时未消费。
- **LLM 工具路由未达标**：召回 100% 后，LLM 对已注入工具的选择仍为 8/11，存在欠调用与误路由，属工具描述/指令的 prompt 工程问题，与召回机制解耦。
- **关键词正则的否定需显式配置**：否定排除依赖人工维护 `negative_keywords`，覆盖不足时仍可能误触发。
