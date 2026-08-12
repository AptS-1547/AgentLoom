# Gateway 会话亲和调度 A/B 压测

> 日期：2026-08-12  
> 状态：实验记录  
> 构建：Windows x64 Release，Visual Studio 2026/v145，MSVC 19.50。  
> 工具：`gateway_session_affinity_bench`。

## 1. 测试目标

本测试只量化 ThreadPool 调度层消除同 session 多 worker 等锁后的变化，不包含 HTTP、Redis、
BERT、LLM 或网络波动。现有 `gateway_concurrency_benchmark.py` 的每个虚拟用户会等待一轮完整
响应后再发送下一轮，同一 session 不产生并发任务，不能稳定复现本次修复的锁竞争。

## 2. 负载

- 每池 4 个 worker；
- 一个热点 session，连续提交 48 项任务；
- 12 个正常 session，每个提交 4 项，共 48 项；
- 总任务数 96，queue capacity 96；
- 默认 FIFO 基线仍用 session mutex 模拟当前 SessionManager 状态保护；
- affinity 场景使用 `GatewaySessionAffinityScheduler`，同 key 最多一个 running；
- Compute 场景 task sleep 2ms；IO 场景 task sleep 8ms；
- 每个场景运行 5 轮，表格取算术平均；
- 所有场景拒绝数均为 0。

Windows `sleep_for(2ms/8ms)` 受系统调度粒度影响，因此绝对耗时不能解释为生产业务 SLA。
同一进程、同一负载下的 FIFO/affinity 相对变化用于判断调度公平性。

## 3. 五轮均值

| Pool | Scheduler | 全任务平均 queue wait | 全任务 P95 | 正常 session 平均 | 正常 session P95 | 正常 session P99 | tasks/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compute | default FIFO | 553.458 ms | 871.301 ms | 793.946 ms | 871.313 ms | 886.884 ms | 106.352 |
| Compute | session affinity | 286.343 ms | 770.811 ms | 93.839 ms | 184.055 ms | 184.078 ms | 111.174 |
| IO | default FIFO | 559.390 ms | 878.579 ms | 800.598 ms | 878.594 ms | 893.976 ms | 105.500 |
| IO | session affinity | 290.544 ms | 779.328 ms | 94.207 ms | 185.350 ms | 185.373 ms | 110.092 |

## 4. 相对变化

| Pool | 全任务平均排队 | 正常 session 平均排队 | 正常 session P95 | 正常 session P99 | 吞吐 |
|---|---:|---:|---:|---:|---:|
| Compute | -48.26% | -88.18% | -78.88% | -79.24% | +4.53% |
| IO | -48.06% | -88.23% | -78.90% | -79.26% | +4.35% |

## 5. 结论

本次优化的主要收益是公平性和正常 session 长尾，而不是热点 session 的总执行时间。热点
session 的任务仍然必须串行完成，因此总体吞吐只提升约 4.4%；但其后续任务不再占用其他
worker 等待同一 session mutex，正常 session 的平均 queue wait 下降约 88%，P95/P99 下降约
79%。

该结果支持在 Gateway Compute/IO Pool 中启用 session-affinity scheduler。生产验收仍需使用
真实 Gateway 流量补充以下测试：

- 同一认证用户并发发送多个请求到同一 session；
- 一个 tenant 下多用户、多 session 洪泛；
- LLM/Redis/gRPC 延迟分布下的 Compute/IO queue wait；
- 429/RESOURCE_EXHAUSTED 比例、拒绝原因和管理端 scheduler 快照；
- 长稳运行中的 lane、fairness 和 tenant 计数归零。

## 6. 复现

```powershell
& "C:\Program Files\CMake\bin\cmake.exe" --build `
  build\x64-Release-Tests-v145-refactor `
  --config Release --target gateway_session_affinity_bench --parallel

1..5 | ForEach-Object {
  & build\x64-Release-Tests-v145-refactor\Release\gateway_session_affinity_bench.exe
}
```
