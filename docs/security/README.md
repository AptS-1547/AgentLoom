# Security Documents

本目录保存认证、Secret、输入、日志、协议和依赖安全的公开规范。

## 当前规范

- [Security Engineering Standard](SECURITY_ENGINEERING_STANDARD.md)

## 相关边界

- 浏览器和 Gateway 协议见 [Gateway](../gateway/README.md)；
- 配置、部署和运行时依赖见 [Runtime](../runtime/README.md)；
- 数据隔离和缓存边界见 [Data](../data/README.md)。

Secret 使用环境变量、Secret Provider 或 ignored 文件，不进入开放文档、样例、日志和性能报告。公开错误、内部 `core::Status`、诊断日志和协议 status code 必须分层。
