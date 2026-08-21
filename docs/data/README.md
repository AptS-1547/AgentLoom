# Data, Cache And Memory Documents

本目录保存 Runtime Memory、缓存、Redis 和数据所有权边界。

## 当前设计与参考

- [Conversation Cache and Inference Strategy](CONVERSATION_CACHE_AND_INFERENCE_STRATEGY.md)
- [Redis Compatibility](REDIS_COMPATIBILITY.md)

## 维护原则

- Runtime Memory、业务数据和文档数据必须明确权威所有者；
- Redis/SQLite/PG 路径、连接池、写入顺序和 retention 变化必须同步更新配置、测试和性能报告；
- 本机实际数据库、模型和依赖目录使用 ignored 配置，开放文档只写仓库相对路径或占位符；
- 数据正确性、跨租户隔离和性能结果分开验收。

旧 Redis++ 迁移过程只在 [Archive](../archive/README.md) 保留。
