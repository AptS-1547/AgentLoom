# 通用异步 gRPC Runtime

`agent_grpc_runtime` 是协议无关的静态库，依赖 `agent_core`、gRPC 和 spdlog，
不依赖 AgentLoom 的 Emotion、BERT、media 或 llama.cpp 模块。安装后的下游目标为
`AgentLoom::grpc_runtime`。

## Unary 接入方式

每个服务在自身模块中从生成的 callback mixin 派生，并在 override 中调用
`AsyncGrpcRuntime::StartUnary`。运行时不知道 proto 类型；`Request` 与 `Response`
由模板参数提供。

```cpp
class IMyHandler final
    : public grpc_runtime::IAsyncUnaryRpcHandler<MyRequest, MyResponse> {
public:
    core::Status Handle(const grpc_runtime::AsyncGrpcCallContext& context,
                        const MyRequest& request,
                        MyResponse& response) override;
};

using MyCallbackBase = MyService::WithCallbackMethod_MyRpc<MyService::Service>;

class MyGrpcService final : public MyCallbackBase {
public:
    grpc::ServerUnaryReactor* MyRpc(grpc::CallbackServerContext* context,
                                    const MyRequest* request,
                                    MyResponse* response) override {
        return runtime_->StartUnary(*context, *request, *response,
                                    handler_, "MyRpc");
    }
};
```

仓库中的 `AsyncEmotionGrpcService` 只是这一路径的可运行示例，不是 runtime 的
类型依赖，也不是其他服务的基类。

## 生命周期与并发契约

- `AsyncGrpcRuntime::Create` 使用 `core::Result` 返回配置错误；线程池不能为空，
  `max_inflight_calls` 必须大于零。
- 每个接收的 unary call 先进行无阻塞的原子准入；到达上限时请求以
  `RESOURCE_EXHAUSTED` 结束，业务 handler 不会执行。
- handler 始终提交到传入的 `core::ThreadPool`，不会在 gRPC callback/EventEngine
  线程上执行。提交失败会以 `UNAVAILABLE` 结束，调用不会悬挂。
- `AsyncGrpcCallContext::deadline` 是绝对 `system_clock` 时间点。进入业务前和业务
  返回后都会检查 deadline；传输层也继续由 gRPC 执行 deadline。
- gRPC `OnCancel` 会请求同一 call 的 `std::stop_source` 停止。handler 应协作地检查
  `stop_token` 并尽快返回；取消不是强制终止正在执行的业务代码。
- runtime 用原子状态机保证每个 call 至多调用一次 `Finish`。排队任务使用共享 call
  state，因此在线程池丢弃排队任务前发生取消时，RPC 仍会完成并释放准入。reactor
  在 gRPC `OnDone` 后销毁，调用方不拥有 reactor。
- callback override 中的 context/request/response 裸指针和返回的 reactor 裸指针是
  gRPC 生成接口的固定签名。runtime 不转移前三者的所有权；gRPC 保证其生命周期持续到
  `OnDone`。reactor 采用 gRPC 约定的自持有模式，在 `OnDone` 中唯一销毁，业务 handler
  不接触这些裸指针，只接收引用和 typed `AsyncGrpcCallContext`。
- 当 `ThreadPool::Shutdown(false)` 丢弃尚未开始的任务时，运行时持有的队列 guard
  会以 `UNAVAILABLE` 完成对应 RPC；不会遗留 in-flight 配额或等待中的客户端。
- `Snapshot` 是无锁原子快照，包含 in-flight、accepted、rejected、completed、failed、
  cancelled、deadline exceeded 和 worker submission failure 计数。

当前版本实现了 typed unary 基础。server-streaming 将复用相同的 `AsyncGrpcCallContext`、
准入、取消、status 映射和 snapshot 基础，并额外引入单 writer 与按 event/byte 限额的
outbound queue；在这些队列语义完整前不应把 streaming 伪装为 unary 的变体。
