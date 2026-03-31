/**
 * @file client_test.cpp
 * @brief BERT gRPC 客户端测试
 */

#include <iostream>
#include <random>
#include <chrono>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>

#include "bert_inference.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using bert_inference::BERTInference;
using bert_inference::PredictRequest;
using bert_inference::PredictResponse;
using bert_inference::PredictBatchRequest;
using bert_inference::PredictBatchResponse;

class BERTClient {
public:
    BERTClient(std::shared_ptr<Channel> channel)
        : stub_(BERTInference::NewStub(channel)) {}

    bool Predict(const std::vector<int64_t>& input_ids,
                 const std::vector<int64_t>& attention_mask,
                 const std::vector<float>& personality,
                 PredictResponse* response) {
        PredictRequest request;
        for (auto v : input_ids) request.add_input_ids(v);
        for (auto v : attention_mask) request.add_attention_mask(v);
        for (auto v : personality) request.add_personality(v);

        ClientContext context;
        Status status = stub_->Predict(&context, request, response);

        return status.ok() && response->error().empty();
    }

    bool PredictBatch(size_t batch_size,
                      size_t seq_len,
                      PredictBatchResponse* response) {
        PredictBatchRequest request;

        // 生成随机测试数据
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int64_t> token_dist(0, 21127);
        std::uniform_real_distribution<float> personality_dist(-1.0f, 1.0f);

        for (size_t i = 0; i < batch_size * seq_len; ++i) {
            request.add_input_ids(token_dist(gen));
            request.add_attention_mask(1);
        }
        for (size_t i = 0; i < batch_size * 11; ++i) {
            request.add_personality(personality_dist(gen));
        }
        request.set_batch_size(batch_size);
        request.set_seq_length(seq_len);

        ClientContext context;
        Status status = stub_->PredictBatch(&context, request, response);

        return status.ok() && response->error().empty();
    }

private:
    std::unique_ptr<BERTInference::Stub> stub_;
};

bool TestSingle(BERTClient& client, size_t seq_len) {
    std::cout << "\n=== Single Prediction Test ===" << std::endl;

    std::vector<int64_t> input_ids(seq_len, 101);  // [CLS]
    std::vector<int64_t> attention_mask(seq_len, 1);
    std::vector<float> personality(11, 0.5f);

    PredictResponse response;

    auto start = std::chrono::high_resolution_clock::now();
    bool success = client.Predict(input_ids, attention_mask, personality, &response);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (success) {
        std::cout << "Success! Latency: " << duration.count() << " ms" << std::endl;
        std::cout << "Emotion logits: [";
        for (int i = 0; i < response.emotion_logits_size() && i < 5; ++i) {
            std::cout << response.emotion_logits(i) << " ";
        }
        std::cout << "...]" << std::endl;
        std::cout << "Intensity: " << response.intensity() << std::endl;
    } else {
        std::cout << "Failed: " << response.error() << std::endl;
    }
    return success;
}

bool TestBatch(BERTClient& client, size_t batch_size, size_t seq_len) {
    std::cout << "\n=== Batch Prediction Test (batch=" << batch_size << ") ===" << std::endl;

    PredictBatchResponse response;

    auto start = std::chrono::high_resolution_clock::now();
    bool success = client.PredictBatch(batch_size, seq_len, &response);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (success) {
        double per_sample = static_cast<double>(duration.count()) / batch_size;
        std::cout << "Success! Total: " << duration.count() << " ms"
                  << " (" << per_sample << " ms/sample)" << std::endl;
        std::cout << "Output samples: " << response.emotion_logits_size() / 10 << std::endl;
    } else {
        std::cout << "Failed: " << response.error() << std::endl;
    }
    return success;
}

bool Benchmark(BERTClient& client, size_t iterations, size_t seq_len) {
    std::cout << "\n=== Benchmark (" << iterations << " iterations) ===" << std::endl;

    std::vector<int64_t> input_ids(seq_len, 101);
    std::vector<int64_t> attention_mask(seq_len, 1);
    std::vector<float> personality(11, 0.5f);

    // 预热
    PredictResponse warmup;
    for (int i = 0; i < 5; ++i) {
        if (!client.Predict(input_ids, attention_mask, personality, &warmup)) {
            std::cout << "Warmup failed" << std::endl;
            return false;
        }
    }

    // 正式测试
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        PredictResponse response;
        if (!client.Predict(input_ids, attention_mask, personality, &response)) {
            std::cout << "Benchmark request failed at iteration " << i << std::endl;
            return false;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double avg_ms = static_cast<double>(total_ms) / iterations;

    std::cout << "Total: " << total_ms << " ms" << std::endl;
    std::cout << "Average: " << avg_ms << " ms/call" << std::endl;
    std::cout << "Throughput: " << (1000.0 / avg_ms) << " calls/sec" << std::endl;
    return true;
}

int main(int argc, char** argv) {
    std::string target = "localhost:50051";
    if (argc > 1) {
        target = argv[1];
    }

    std::cout << "Connecting to " << target << "..." << std::endl;

    auto channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    BERTClient client(channel);

    // 测试单条
    const bool single_ok = TestSingle(client, 64);

    // 测试不同 batch 大小
    const bool batch4_ok = TestBatch(client, 4, 64);
    const bool batch8_ok = TestBatch(client, 8, 64);

    // 基准测试
    const bool benchmark_ok = Benchmark(client, 100, 64);

    return (single_ok && batch4_ok && batch8_ok && benchmark_ok) ? 0 : 1;
}
