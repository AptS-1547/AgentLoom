# Chat Prompt KV Cache 复用边界设计

## 目标

主链路 Chat 不应要求完整 prompt 完全相等才复用 KV。合理目标是：

```text
严格验证 runtime/model 边界
→ 在兼容状态中寻找最长的逐 token 相同前缀
→ 恢复该前缀 KV
→ 删除分叉点后的旧状态
→ 只计算新增或变化的 suffix
```

这里的“更松”是候选选择允许不同的完整 prompt；真正恢复的 token 前缀仍必须逐 token、逐位置完全一致。语义相似或向量临近不能直接用于 KV state 复用。

## 当前主链路缺口

当前本地 Chat 路径为：

```text
PersonaRuntime
→ ChatCompletionRequest
→ LocalLlmChatClient::BuildPrompt
→ LocalLlmRequest
→ GenerateVLMSync
→ LlamaRunner::Generate
```

现状限制：

1. `ChatCompletionRequest` 没有 session、request、conversation branch 标识。
2. `LocalLlmChatClient` 没有把 `PersonaRuntime::ChatRequest::session_id` 传到推理端。
3. messages 被自定义 `<role>` 标签拼接为扁平文本，没有使用模型的 canonical chat template。
4. 纯文本请求没有 image bytes；当前 VLM Prompt KV key 要求 image bytes，因此 Chat KV 实际不会启用。
5. `LlamaRunnerPool::Acquire()` 只获取任意空闲 slot，没有按 session 或 token LCP 选择最合适的 runner。
6. 当前 Prompt KV entry 只保存 state 和 prefix token 数，没有保存 token 序列、message boundary、chat template/runtime fingerprint。

## 不可放宽的兼容边界

以下任一项变化时不得复用旧 KV：

- 模型文件与权重 fingerprint；
- tokenizer/vocabulary 与 special token 定义；
- chat template 及模板版本；
- RoPE、context、SWA/recurrent memory 等影响位置和状态布局的参数；
- LoRA、aLoRA、control vector 或其他 adapter 集合与顺序；
- system prompt、persona 基础定义和安全策略版本；
- tool/function schema 及其序列化顺序；
- 多模态前缀中的图片、音频等输入 hash；
- llama.cpp state ABI/schema 和构建版本；
- 需要隔离的 tenant、用户、persona 或权限 scope。

生成参数中的 temperature、top-p、top-k、max tokens 不影响已经计算出的 prompt KV，可以变化；但 grammar、adapter 和会改变 prompt token 序列的参数必须纳入兼容 fingerprint。

## 可复用边界

### 1. 全局静态前缀

适合缓存：

- 固定 system prompt；
- 安全规范；
- 不含用户私有信息的公共工具定义；
- 固定 persona 模板。

可跨 session 复用，但必须按 tenant/persona/权限 scope 隔离。建议只在至少 64～128 个 token 时建立 checkpoint，避免恢复开销大于节省的 prompt eval。

### 2. Session 对话前缀

同一 session 的后续轮次通常包含上一轮的完整 messages：

```text
system + turn 1 + turn 2
system + turn 1 + turn 2 + turn 3
```

可以恢复上一轮结束处的 KV，只计算新一轮 user message 和 assistant marker。该路径应优先保持 runner slot affinity；slot 不可用时再从共享 KV backend 恢复。

### 3. 会话分叉

用户重试、编辑上一条消息或从历史节点创建分支时：

```text
旧：system + A + B + C
新：system + A + B + D
```

复用 `system + A + B` 的最长公共 token 前缀，在分叉点调用 sequence remove，删除旧 suffix，再计算 `D`。

### 4. 动态上下文注入

记忆、视觉事件、检索结果和时间信息应尽量放在 prompt 后部。若高频变化内容插入 system prompt 中间，会破坏后续所有 token 的 LCP，显著降低 KV 命中率。

推荐 prompt 结构：

```text
稳定 system/persona/tools
→ 稳定历史 messages
→ 本轮 memory/RAG/vision context
→ 当前 user message
→ assistant marker
```

如果业务语义允许，可对注入内容做稳定排序和 canonical JSON 序列化，避免仅因字段顺序变化导致 token 前缀失效。

## 建议的数据结构

```cpp
struct ChatKvFingerprint {
    std::string model;
    std::string tokenizer;
    std::string chat_template;
    std::string runtime_abi;
    std::string adapter_set;
    std::string tenant_scope;
    std::string persona_revision;
    std::string tool_schema;
};

struct ChatKvCheckpoint {
    ChatKvFingerprint fingerprint;
    std::vector<llama_token> tokens;
    std::vector<std::uint32_t> message_boundaries;
    std::vector<std::uint8_t> state;
    std::int32_t prefix_tokens = 0;
};
```

业务接口仍应使用明确类型，不传递裸 `void*`。state 由 RAII entry 持有，backend 返回 `core::Result`，失败通过 logger 记录后回退到重新计算。

## 索引和选择策略

不建议只按整段 SHA-256 建索引。建议分两级：

1. fingerprint bucket：先排除 runtime/model/scope 不兼容状态；
2. token prefix radix tree 或分段 hash：在 bucket 内寻找最长公共 token 前缀。

第一版可以只在 message boundary 建 checkpoint，并保存每个 boundary 的 rolling token hash：

```text
hash(tokens[0..message_1_end])
hash(tokens[0..message_2_end])
hash(tokens[0..message_3_end])
```

从最新 boundary 向前查找第一个命中项，随后仍比较实际 token，不能只相信 hash。

候选评分建议综合：

```text
可复用 token 数
- state restore 延迟折算 token
- runner 排队代价
- state 搬运字节成本
```

只有预计收益为正时才恢复持久化 state；同 runner 的热 KV 可以使用更低门槛。

## 多 runner 层级

建议分为三级：

```text
L0：runner slot 内仍驻留的 session/LCP KV
L1：进程内共享的序列 state
L2：可选 Redis/文件持久化 state
```

优先级：

```text
同 session 热 slot
→ 其他 slot 中更长 LCP
→ 进程内 checkpoint
→ 跨进程 checkpoint
→ fresh prompt eval
```

Redis 更适合存索引和跨进程冷状态，不适合每轮无条件搬运几十 MiB state。热对话应优先保持 slot affinity。

## 与 Answer Cache 的顺序

主链路目前还有 Answer Cache，它会直接绕过 LLM。若目标是降低机械重复并保留重新生成能力，推荐策略与 VLM 一致：

```text
Chat KV/LCP 可复用
→ 恢复 KV 并重新生成
→ 无可用 KV 时再按业务策略查询 Answer Cache
→ fresh LLM
```

Answer Cache 仍可保留为：

- 极低延迟模式；
- 模型不可用或超时 fallback；
- 明确要求确定性答案的任务；
- 高置信度、低风险的事实型请求。

## 指标

至少输出：

- `prompt_tokens_total`；
- `prompt_tokens_cached`；
- `prompt_tokens_processed`；
- `kv_hit_level`：slot/process/persistent/miss；
- `kv_restore_ms`；
- `prompt_eval_ms`；
- `state_bytes_loaded/stored`；
- `lcp_tokens` 和 `lcp_ratio`；
- slot affinity hit rate；
- 因 fingerprint/scope 不兼容被拒绝的次数。

最终性能评价应使用：

```text
净节省 = fresh prompt eval 时间
       - KV Probe 时间
       - state restore/搬运时间
       - runner affinity 额外排队时间
```

## 推荐实施顺序

1. 给 `ChatCompletionRequest` 和 `LocalLlmRequest` 贯穿 session/request/branch 标识。
2. 用模型 canonical chat template 替换自定义 `<role>` 拼接，并暴露 token 序列。
3. 实现单 runner、同 session 的热 KV 增量追加。
4. 实现编辑/重试场景的 token LCP 截断与 suffix 重算。
5. 为 runner pool 增加 session/LCP affinity 选择。
6. 增加 message-boundary 进程内 checkpoint。
7. 最后再评估 Redis 跨进程 state 是否具有正收益。

第一阶段不应直接实现“语义相似 KV”：只有逐 token 相同的前缀才能安全复用。Vector Cache 可以帮助答案或上下文检索，但不能替代 token LCP 验证。
