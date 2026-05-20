#include "hf_tokenizer.h"

#include "hf_tokenizers_capi.h"

#include <cstring>
#include <string>
#include <utility>

namespace vector {

namespace {

core::ErrorCode MapStatus(std::int32_t code) noexcept {
    switch (code) {
        case HF_TOKENIZER_OK:                  return core::ErrorCode::Ok;
        case HF_TOKENIZER_ERR_NULL_POINTER:    return core::ErrorCode::InvalidArgument;
        case HF_TOKENIZER_ERR_INVALID_ARG:     return core::ErrorCode::InvalidArgument;
        case HF_TOKENIZER_ERR_INVALID_UTF8:    return core::ErrorCode::InvalidArgument;
        case HF_TOKENIZER_ERR_IO:              return core::ErrorCode::Unavailable;
        case HF_TOKENIZER_ERR_TOKENIZER_LOAD:  return core::ErrorCode::FailedPrecondition;
        case HF_TOKENIZER_ERR_ENCODE_FAILED:   return core::ErrorCode::InternalError;
        case HF_TOKENIZER_ERR_OUT_OF_BOUNDS:   return core::ErrorCode::InvalidArgument;
        case HF_TOKENIZER_ERR_PANIC:           return core::ErrorCode::InternalError;
        case HF_TOKENIZER_ERR_INTERNAL:        return core::ErrorCode::InternalError;
        default:                               return core::ErrorCode::Unknown;
    }
}

std::string FetchLastErrorMessage() {
    const char* msg = hf_tokenizers_last_error_message();
    return msg != nullptr ? std::string(msg) : std::string{};
}

core::Status MakeStatus(std::int32_t code, std::string_view context) {
    if (code == HF_TOKENIZER_OK) {
        return core::Status::Ok();
    }
    std::string message;
    message.reserve(context.size() + 64);
    message.append(context);
    message.append(" (code=");
    message.append(std::to_string(code));
    message.append(")");
    std::string detail = FetchLastErrorMessage();
    if (!detail.empty()) {
        message.append(": ");
        message.append(detail);
    }
    return core::Status(MapStatus(code), std::move(message));
}

hf_encode_options_t ToCOptions(const EncodeOptions& opts) noexcept {
    hf_encode_options_t out{};
    out.max_length = opts.max_length;
    out.truncation = opts.truncation ? 1 : 0;
    out.padding = opts.padding ? 1 : 0;
    out.pad_to_longest_in_batch = opts.pad_to_longest_in_batch ? 1 : 0;
    out.add_special_tokens = opts.add_special_tokens ? 1 : 0;
    for (auto& b : out.reserved) {
        b = 0;
    }
    return out;
}

hf_byte_span_t ToSpan(std::string_view sv) noexcept {
    hf_byte_span_t span{};
    span.ptr = reinterpret_cast<const std::uint8_t*>(sv.data());
    span.len = sv.size();
    return span;
}

struct BatchDeleter {
    void operator()(hf_tokenized_batch_t* p) const noexcept {
        if (p != nullptr) {
            hf_tokenized_batch_destroy(p);
        }
    }
};

using BatchPtr = std::unique_ptr<hf_tokenized_batch_t, BatchDeleter>;

core::Result<TokenizedBatch> MaterializeBatch(BatchPtr batch) {
    TokenizedBatch out;
    std::size_t batch_size = 0;
    std::size_t sequence_length = 0;

    if (auto code = hf_tokenized_batch_batch_size(batch.get(), &batch_size); code != HF_TOKENIZER_OK) {
        return MakeStatus(code, "hf_tokenized_batch_batch_size");
    }
    if (auto code = hf_tokenized_batch_sequence_length(batch.get(), &sequence_length); code != HF_TOKENIZER_OK) {
        return MakeStatus(code, "hf_tokenized_batch_sequence_length");
    }

    const std::int64_t* ids_ptr = nullptr;
    const std::int64_t* mask_ptr = nullptr;
    const std::int64_t* types_ptr = nullptr;

    if (auto code = hf_tokenized_batch_input_ids(batch.get(), &ids_ptr); code != HF_TOKENIZER_OK) {
        return MakeStatus(code, "hf_tokenized_batch_input_ids");
    }
    if (auto code = hf_tokenized_batch_attention_mask(batch.get(), &mask_ptr); code != HF_TOKENIZER_OK) {
        return MakeStatus(code, "hf_tokenized_batch_attention_mask");
    }
    if (auto code = hf_tokenized_batch_token_type_ids(batch.get(), &types_ptr); code != HF_TOKENIZER_OK) {
        return MakeStatus(code, "hf_tokenized_batch_token_type_ids");
    }

    const std::size_t total = batch_size * sequence_length;
    out.batch_size = batch_size;
    out.sequence_length = sequence_length;
    if (total > 0) {
        if (ids_ptr == nullptr || mask_ptr == nullptr || types_ptr == nullptr) {
            return core::Status(core::ErrorCode::InternalError,
                                "tokenizer returned null buffer with non-zero size");
        }
        out.input_ids.assign(ids_ptr, ids_ptr + total);
        out.attention_mask.assign(mask_ptr, mask_ptr + total);
        out.token_type_ids.assign(types_ptr, types_ptr + total);
    }
    return out;
}

} // namespace

struct HfTokenizer::Impl {
    hf_tokenizer_t* handle = nullptr;
    std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();

    ~Impl() {
        if (handle != nullptr) {
            hf_tokenizer_destroy(handle);
            handle = nullptr;
        }
    }

    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) noexcept = default;
    Impl& operator=(Impl&&) noexcept = default;
};

HfTokenizer::HfTokenizer() noexcept = default;
HfTokenizer::~HfTokenizer() noexcept = default;
HfTokenizer::HfTokenizer(HfTokenizer&&) noexcept = default;
HfTokenizer& HfTokenizer::operator=(HfTokenizer&&) noexcept = default;

bool HfTokenizer::valid() const noexcept {
    return impl_ != nullptr && impl_->handle != nullptr;
}

int HfTokenizer::AbiVersion() noexcept {
    return static_cast<int>(hf_tokenizers_abi_version());
}

core::Result<HfTokenizer> HfTokenizer::Clone() const {
    if (!valid()) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "HfTokenizer::Clone: source handle is not valid");
    }

    hf_tokenizer_t* raw = nullptr;
    auto code = hf_tokenizer_clone(impl_->handle, &raw);
    if (code != HF_TOKENIZER_OK) {
        return MakeStatus(code, "hf_tokenizer_clone");
    }

    HfTokenizer out;
    out.impl_ = std::make_unique<Impl>();
    out.impl_->handle = raw;
    return out;
}

core::Result<HfTokenizer> HfTokenizer::LoadFromFile(const std::filesystem::path& path) {
    std::string utf8 = path.u8string().empty()
        ? path.string()
        : std::string(reinterpret_cast<const char*>(path.u8string().data()),
                      path.u8string().size());

    hf_byte_span_t span{};
    span.ptr = reinterpret_cast<const std::uint8_t*>(utf8.data());
    span.len = utf8.size();

    hf_tokenizer_t* raw = nullptr;
    auto code = hf_tokenizer_create_from_file(span, &raw);
    if (code != HF_TOKENIZER_OK) {
        return MakeStatus(code, "hf_tokenizer_create_from_file");
    }

    HfTokenizer out;
    out.impl_ = std::make_unique<Impl>();
    out.impl_->handle = raw;
    return out;
}

core::Result<TokenizedBatch> HfTokenizer::Encode(
    std::string_view text,
    const EncodeOptions& options) const {
    if (!valid()) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "HfTokenizer::Encode: handle is not valid");
    }

    std::lock_guard<std::mutex> lock(*impl_->mutex);

    hf_encode_options_t c_opts = ToCOptions(options);
    hf_byte_span_t span = ToSpan(text);

    hf_tokenized_batch_t* raw = nullptr;
    auto code = hf_tokenizer_encode(impl_->handle, span, &c_opts, &raw);
    BatchPtr batch(raw);
    if (code != HF_TOKENIZER_OK) {
        return MakeStatus(code, "hf_tokenizer_encode");
    }
    return MaterializeBatch(std::move(batch));
}

core::Result<TokenizedBatch> HfTokenizer::EncodeBatch(
    std::span<const std::string_view> texts,
    const EncodeOptions& options) const {
    if (!valid()) {
        return core::Status(core::ErrorCode::FailedPrecondition,
                            "HfTokenizer::EncodeBatch: handle is not valid");
    }
    if (texts.empty()) {
        return core::Status(core::ErrorCode::InvalidArgument,
                            "HfTokenizer::EncodeBatch: texts is empty");
    }

    std::vector<hf_byte_span_t> spans;
    spans.reserve(texts.size());
    for (const auto& t : texts) {
        spans.push_back(ToSpan(t));
    }

    std::lock_guard<std::mutex> lock(*impl_->mutex);

    hf_encode_options_t c_opts = ToCOptions(options);

    hf_tokenized_batch_t* raw = nullptr;
    auto code = hf_tokenizer_encode_batch(
        impl_->handle, spans.data(), spans.size(), &c_opts, &raw);
    BatchPtr batch(raw);
    if (code != HF_TOKENIZER_OK) {
        return MakeStatus(code, "hf_tokenizer_encode_batch");
    }
    return MaterializeBatch(std::move(batch));
}

} // namespace vector
