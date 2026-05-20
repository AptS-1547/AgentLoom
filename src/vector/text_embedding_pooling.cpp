#include "text_embedding_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vector::pooling {

core::Status MeanPool(
    const float* last_hidden_state,
    const std::int64_t* attention_mask,
    std::size_t batch_size,
    std::size_t sequence_length,
    std::size_t hidden_size,
    float* out_embeddings) {

    if (last_hidden_state == nullptr || attention_mask == nullptr || out_embeddings == nullptr) {
        return core::Status(core::ErrorCode::InvalidArgument, "MeanPool: null buffer");
    }
    if (batch_size == 0 || sequence_length == 0 || hidden_size == 0) {
        return core::Status(core::ErrorCode::InvalidArgument, "MeanPool: zero dimension");
    }

    for (std::size_t b = 0; b < batch_size; ++b) {
        float* out_row = out_embeddings + b * hidden_size;
        std::fill_n(out_row, hidden_size, 0.0f);

        std::int64_t mask_sum = 0;
        for (std::size_t t = 0; t < sequence_length; ++t) {
            const std::int64_t m = attention_mask[b * sequence_length + t];
            if (m == 0) {
                continue;
            }
            mask_sum += 1;
            const float* token_row =
                last_hidden_state + (b * sequence_length + t) * hidden_size;
            for (std::size_t h = 0; h < hidden_size; ++h) {
                out_row[h] += token_row[h];
            }
        }

        const float divisor = static_cast<float>(std::max<std::int64_t>(mask_sum, 1));
        for (std::size_t h = 0; h < hidden_size; ++h) {
            out_row[h] /= divisor;
        }
    }
    return core::Status::Ok();
}

core::Status ClsPool(
    const float* last_hidden_state,
    std::size_t batch_size,
    std::size_t sequence_length,
    std::size_t hidden_size,
    float* out_embeddings) {

    if (last_hidden_state == nullptr || out_embeddings == nullptr) {
        return core::Status(core::ErrorCode::InvalidArgument, "ClsPool: null buffer");
    }
    if (batch_size == 0 || sequence_length == 0 || hidden_size == 0) {
        return core::Status(core::ErrorCode::InvalidArgument, "ClsPool: zero dimension");
    }

    for (std::size_t b = 0; b < batch_size; ++b) {
        const float* token0 = last_hidden_state + b * sequence_length * hidden_size;
        float* out_row = out_embeddings + b * hidden_size;
        std::memcpy(out_row, token0, hidden_size * sizeof(float));
    }
    return core::Status::Ok();
}

void L2NormalizeRows(float* data, std::size_t batch_size, std::size_t dimension) noexcept {
    if (data == nullptr || batch_size == 0 || dimension == 0) {
        return;
    }
    for (std::size_t b = 0; b < batch_size; ++b) {
        float* row = data + b * dimension;
        double sq_sum = 0.0;
        for (std::size_t d = 0; d < dimension; ++d) {
            sq_sum += static_cast<double>(row[d]) * static_cast<double>(row[d]);
        }
        const float norm = static_cast<float>(std::sqrt(sq_sum));
        if (norm <= 1e-12f) {
            continue;
        }
        const float inv = 1.0f / norm;
        for (std::size_t d = 0; d < dimension; ++d) {
            row[d] *= inv;
        }
    }
}

} // namespace vector::pooling
