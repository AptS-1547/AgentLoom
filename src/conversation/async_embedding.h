#pragma once

// 协程批量 embedding：把逐条回调式的 EmbeddingBatchCoordinator 桥接成一次 co_await。
// 复用 EmbeddingBatchCoordinator 的批量队列 + 共享模型 + max_inflight_batches GPU 配额 +
// 按序回推，CPU 协程在 GPU 执行期间挂起（释放线程），完成后按序组装成 Batch。

#include "dialogue_segmenter.h"

#include "../core/coroutine_awaitable.h"
#include "../core/coroutine_task.h"
#include "../vector/embedding_batch_coordinator.h"

#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace agent::conversation {

inline core::async::task<core::Result<ITextEmbeddingProvider::Batch>> EmbedBatchAsync(
    vector::EmbeddingBatchCoordinator& coordinator,
    std::span<const std::string_view> texts,
    std::string_view session_id = "embed-batch") {
    using Batch = ITextEmbeddingProvider::Batch;
    using Completion = std::function<void(core::Result<Batch>)>;

    const std::size_t count = texts.size();
    if (count == 0) {
        co_return Batch{};
    }

    // 用 CallbackAwaiter 等「全部 N 条完成」这一个信号；Collector 把 N 个单条 completion
    // 聚合成一个 batch completion。
    auto batch_result = co_await core::async::CallbackAwaiter<Batch>(
        [&](Completion completion) -> core::Status {
            struct Collector {
                std::mutex mutex;
                std::size_t remaining;
                std::size_t dimension = 0;
                std::vector<std::vector<float>> embeddings;  // 按 index 存成功结果
                std::optional<core::Status> first_error;
                Completion completion;

                void OnResult(std::size_t index, core::Result<std::vector<float>> result) {
                    std::optional<core::Result<Batch>> batch_result;
                    bool finished = false;
                    {
                        std::lock_guard lock(mutex);
                        if (result.ok()) {
                            embeddings[index] = std::move(result).value();
                            if (dimension == 0) {
                                dimension = embeddings[index].size();
                            }
                        } else if (!first_error) {
                            first_error = result.status();
                        }
                        --remaining;
                        if (remaining == 0) {
                            finished = true;
                            batch_result.emplace(Assemble());
                        }
                    }
                    if (finished) {
                        completion(std::move(*batch_result));
                    }
                }

                core::Result<Batch> Assemble() {
                    if (first_error) return *first_error;
                    Batch batch;
                    batch.batch_size = embeddings.size();
                    batch.dimension = dimension;
                    batch.embeddings.reserve(embeddings.size() * dimension);
                    for (auto& embedding : embeddings) {
                        batch.embeddings.insert(batch.embeddings.end(),
                                                embedding.begin(), embedding.end());
                    }
                    return batch;
                }
            };

            auto collector = std::make_shared<Collector>();
            collector->remaining = count;
            collector->embeddings.resize(count);
            collector->completion = std::move(completion);

            for (std::size_t index = 0; index < count; ++index) {
                vector::EmbeddingBatchRequest request;
                request.session_id = std::string(session_id);
                request.text = std::string(texts[index]);
                request.completion = [collector, index](
                                        core::Result<std::vector<float>> result) {
                    collector->OnResult(index, std::move(result));
                };
                const auto status = coordinator.Submit(std::move(request));
                if (!status.ok()) {
                    // 提交失败：该条视为失败，补一次完成。
                    collector->OnResult(index, status);
                }
            }
            return core::Status::Ok();
        });

    co_return batch_result;
}

} // namespace agent::conversation
