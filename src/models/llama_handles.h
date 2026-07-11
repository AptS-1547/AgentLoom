#pragma once

#include "unique_handle.h"

#include <llama.h>
#include <mtmd.h>

#include <cstdint>

namespace core {

template <>
struct ResourceTraits<llama_batch> {
    using handle_type = llama_batch;

    static handle_type invalid() noexcept {
        return {};
    }

    static bool valid(const handle_type& handle) noexcept {
        return handle.token != nullptr ||
               handle.embd != nullptr ||
               handle.pos != nullptr ||
               handle.n_seq_id != nullptr ||
               handle.seq_id != nullptr ||
               handle.logits != nullptr;
    }

    static void close(handle_type& handle) noexcept {
        llama_batch_free(handle);
    }
};

template <>
struct ResourceTraits<llama_model> {
    using handle_type = llama_model*;

    static handle_type invalid() noexcept {
        return nullptr;
    }

    static bool valid(handle_type handle) noexcept {
        return handle != nullptr;
    }

    static void close(handle_type handle) noexcept {
        llama_free_model(handle);
    }
};

template <>
struct ResourceTraits<llama_context> {
    using handle_type = llama_context*;

    static handle_type invalid() noexcept {
        return nullptr;
    }

    static bool valid(handle_type handle) noexcept {
        return handle != nullptr;
    }

    static void close(handle_type handle) noexcept {
        llama_free(handle);
    }
};

template <>
struct ResourceTraits<llama_sampler> {
    using handle_type = llama_sampler*;

    static handle_type invalid() noexcept {
        return nullptr;
    }

    static bool valid(handle_type handle) noexcept {
        return handle != nullptr;
    }

    static void close(handle_type handle) noexcept {
        llama_sampler_free(handle);
    }
};

template <>
struct ResourceTraits<mtmd_context> {
    using handle_type = mtmd_context*;

    static handle_type invalid() noexcept {
        return nullptr;
    }

    static bool valid(handle_type handle) noexcept {
        return handle != nullptr;
    }

    static void close(handle_type handle) noexcept {
        mtmd_free(handle);
    }
};

template <>
struct ResourceTraits<mtmd_bitmap> {
    using handle_type = mtmd_bitmap*;

    static handle_type invalid() noexcept {
        return nullptr;
    }

    static bool valid(handle_type handle) noexcept {
        return handle != nullptr;
    }

    static void close(handle_type handle) noexcept {
        mtmd_bitmap_free(handle);
    }
};

template <>
struct ResourceTraits<mtmd_input_chunks> {
    using handle_type = mtmd_input_chunks*;

    static handle_type invalid() noexcept {
        return nullptr;
    }

    static bool valid(handle_type handle) noexcept {
        return handle != nullptr;
    }

    static void close(handle_type handle) noexcept {
        mtmd_input_chunks_free(handle);
    }
};

template <>
struct ResourceTraits<mtmd_batch> {
    using handle_type = mtmd_batch*;

    static handle_type invalid() noexcept {
        return nullptr;
    }

    static bool valid(handle_type handle) noexcept {
        return handle != nullptr;
    }

    static void close(handle_type handle) noexcept {
        mtmd_batch_free(handle);
    }
};

} // namespace core

namespace llm {

using LlamaBatchHandle = core::UniqueHandle<llama_batch>;
using LlamaModelHandle = core::UniqueHandle<llama_model>;
using LlamaContextHandle = core::UniqueHandle<llama_context>;
using LlamaSamplerHandle = core::UniqueHandle<llama_sampler>;
using MtmdContextHandle = core::UniqueHandle<mtmd_context>;
using MtmdBitmapHandle = core::UniqueHandle<mtmd_bitmap>;
using MtmdInputChunksHandle = core::UniqueHandle<mtmd_input_chunks>;
using MtmdBatchHandle = core::UniqueHandle<mtmd_batch>;

inline LlamaBatchHandle MakeLlamaBatch(int32_t n_tokens, int32_t embd = 0, int32_t n_seq_max = 1) noexcept {
    return core::make_unique_handle<llama_batch>(llama_batch_init(n_tokens, embd, n_seq_max));
}

inline MtmdInputChunksHandle MakeMtmdInputChunks() noexcept {
    return core::make_unique_handle<mtmd_input_chunks>(mtmd_input_chunks_init());
}

inline MtmdBatchHandle MakeMtmdBatch(mtmd_context* ctx) noexcept {
    return core::make_unique_handle<mtmd_batch>(mtmd_batch_init(ctx));
}

} // namespace llm
