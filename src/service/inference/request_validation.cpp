#include "request_validation.h"
#include "text_validation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <string>
#include <utility>

namespace request_validation {
namespace {

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool ConstantTimeEquals(std::string_view lhs, std::string_view rhs) {
    const size_t max_size = std::max(lhs.size(), rhs.size());
    unsigned char diff = static_cast<unsigned char>(lhs.size() ^ rhs.size());
    for (size_t i = 0; i < max_size; ++i) {
        const unsigned char l = i < lhs.size() ? static_cast<unsigned char>(lhs[i]) : 0;
        const unsigned char r = i < rhs.size() ? static_cast<unsigned char>(rhs[i]) : 0;
        diff |= static_cast<unsigned char>(l ^ r);
    }
    return diff == 0;
}

uint16_t ReadBe16(const uint8_t* data) {
    return static_cast<uint16_t>((static_cast<uint16_t>(data[0]) << 8) | data[1]);
}

uint32_t ReadBe32(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
           static_cast<uint32_t>(data[3]);
}

uint16_t ReadLe16(const uint8_t* data) {
    return static_cast<uint16_t>(data[0] | (static_cast<uint16_t>(data[1]) << 8));
}

uint32_t ReadLe24(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16);
}

uint32_t ReadLe32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

int32_t ReadLe32Signed(const uint8_t* data) {
    const uint32_t value = ReadLe32(data);
    int32_t signed_value = 0;
    static_assert(sizeof(signed_value) == sizeof(value));
    std::memcpy(&signed_value, &value, sizeof(value));
    return signed_value;
}

bool BytesEqual(const uint8_t* data, std::string_view expected) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (data[i] != static_cast<uint8_t>(expected[i])) {
            return false;
        }
    }
    return true;
}

struct ImageInfo {
    std::string format;
    uint32_t width = 0;
    uint32_t height = 0;
};

bool ParsePng(const uint8_t* data, size_t size, ImageInfo& info) {
    static constexpr uint8_t signature[] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    if (size < 33 || !std::equal(std::begin(signature), std::end(signature), data)) {
        return false;
    }
    if (!BytesEqual(data + 12, "IHDR")) {
        return false;
    }
    if (ReadBe32(data + 8) != 13) {
        return false;
    }
    info.format = "png";
    info.width = ReadBe32(data + 16);
    info.height = ReadBe32(data + 20);
    return true;
}

bool IsJpegSofMarker(uint8_t marker) {
    return marker == 0xC0 || marker == 0xC1 || marker == 0xC2 ||
           marker == 0xC3 || marker == 0xC5 || marker == 0xC6 ||
           marker == 0xC7 || marker == 0xC9 || marker == 0xCA ||
           marker == 0xCB || marker == 0xCD || marker == 0xCE ||
           marker == 0xCF;
}

bool IsJpegStandaloneMarker(uint8_t marker) {
    return marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7);
}

bool ParseJpeg(const uint8_t* data, size_t size, ImageInfo& info) {
    if (size < 4 || data[0] != 0xFF || data[1] != 0xD8) {
        return false;
    }

    size_t pos = 2;
    while (pos + 1 < size) {
        if (data[pos] != 0xFF) {
            ++pos;
            continue;
        }

        while (pos < size && data[pos] == 0xFF) {
            ++pos;
        }
        if (pos >= size) {
            break;
        }

        const uint8_t marker = data[pos++];
        if (marker == 0xD9 || marker == 0xDA) {
            break;
        }
        if (IsJpegStandaloneMarker(marker)) {
            continue;
        }
        if (pos + 2 > size) {
            return false;
        }

        const uint16_t segment_length = ReadBe16(data + pos);
        if (segment_length < 2) {
            return false;
        }
        if (pos + segment_length > size) {
            return false;
        }

        if (IsJpegSofMarker(marker)) {
            if (segment_length < 7) {
                return false;
            }
            info.format = "jpeg";
            info.height = ReadBe16(data + pos + 3);
            info.width = ReadBe16(data + pos + 5);
            return true;
        }

        pos += segment_length;
    }

    return false;
}

bool ParseGif(const uint8_t* data, size_t size, ImageInfo& info) {
    if (size < 10) {
        return false;
    }
    if (!BytesEqual(data, "GIF87a") && !BytesEqual(data, "GIF89a")) {
        return false;
    }
    info.format = "gif";
    info.width = ReadLe16(data + 6);
    info.height = ReadLe16(data + 8);
    return true;
}

bool ParseBmp(const uint8_t* data, size_t size, ImageInfo& info) {
    if (size < 26 || data[0] != 'B' || data[1] != 'M') {
        return false;
    }

    const uint32_t dib_size = ReadLe32(data + 14);
    if (dib_size == 12) {
        info.format = "bmp";
        info.width = ReadLe16(data + 18);
        info.height = ReadLe16(data + 20);
        return true;
    }

    if (dib_size >= 40 && size >= 26) {
        const int32_t width = ReadLe32Signed(data + 18);
        const int32_t height = ReadLe32Signed(data + 22);
        if (width <= 0 || height == 0 || height == std::numeric_limits<int32_t>::min()) {
            return false;
        }
        info.format = "bmp";
        info.width = static_cast<uint32_t>(width);
        info.height = static_cast<uint32_t>(height < 0 ? -height : height);
        return true;
    }

    return false;
}

bool ParseWebp(const uint8_t* data, size_t size, ImageInfo& info) {
    if (size < 30 || !BytesEqual(data, "RIFF") || !BytesEqual(data + 8, "WEBP")) {
        return false;
    }

    if (BytesEqual(data + 12, "VP8X")) {
        info.format = "webp";
        info.width = ReadLe24(data + 24) + 1;
        info.height = ReadLe24(data + 27) + 1;
        return true;
    }

    if (BytesEqual(data + 12, "VP8 ")) {
        if (data[23] != 0x9D || data[24] != 0x01 || data[25] != 0x2A) {
            return false;
        }
        info.format = "webp";
        info.width = ReadLe16(data + 26) & 0x3FFF;
        info.height = ReadLe16(data + 28) & 0x3FFF;
        return true;
    }

    if (BytesEqual(data + 12, "VP8L") && size >= 25) {
        if (data[20] != 0x2F) {
            return false;
        }
        const uint32_t bits = ReadLe32(data + 21);
        info.format = "webp";
        info.width = (bits & 0x3FFF) + 1;
        info.height = ((bits >> 14) & 0x3FFF) + 1;
        return true;
    }

    return false;
}

bool ParseImageInfo(std::string_view bytes, ImageInfo& info) {
    if (bytes.empty()) {
        return false;
    }

    const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
    const size_t size = bytes.size();
    return ParsePng(data, size, info) ||
           ParseJpeg(data, size, info) ||
           ParseGif(data, size, info) ||
           ParseBmp(data, size, info) ||
           ParseWebp(data, size, info);
}

ValidationResult ValidateText(std::string_view text, size_t max_bytes, bool require_non_empty, std::string_view name) {
    if (require_non_empty && text.empty()) {
        return Fail(std::string(name) + " is empty");
    }
    if (text.size() > max_bytes) {
        return Fail(std::string(name) + " exceeds byte limit");
    }
    if (!core::IsValidUtf8(text)) {
        return Fail(std::string(name) + " is not valid UTF-8");
    }
    if (core::HasInvalidTextControl(text)) {
        return Fail(std::string(name) + " contains invalid control characters");
    }
    return Ok();
}

bool IsSafeMetadataChar(unsigned char c) {
    return std::isalnum(c) ||
           c == '_' ||
           c == '-' ||
           c == '.' ||
           c == ':' ||
           c == '@';
}

ValidationResult ValidateMetadataText(std::string_view text, size_t max_bytes, std::string_view name) {
    if (text.empty()) {
        return Ok();
    }
    if (text.size() > max_bytes) {
        return Fail(std::string(name) + " exceeds byte limit");
    }
    for (unsigned char c : text) {
        if (!IsSafeMetadataChar(c)) {
            return Fail(std::string(name) + " contains invalid characters");
        }
    }
    return Ok();
}

ValidationResult ValidateImageData(std::string_view bytes, const RequestLimits& limits) {
    if (bytes.empty()) {
        return Fail("image_data is empty");
    }
    if (bytes.size() > limits.max_image_bytes) {
        return Fail("image_data exceeds byte limit");
    }

    ImageInfo info;
    if (!ParseImageInfo(bytes, info)) {
        return Fail("image_data has unsupported or invalid image header");
    }
    if (info.width == 0 || info.height == 0) {
        return Fail("image dimensions are empty");
    }
    if (info.width > limits.max_image_width || info.height > limits.max_image_height) {
        return Fail("image dimensions exceed limit");
    }
    if (info.height != 0 && info.width > std::numeric_limits<size_t>::max() / info.height) {
        return Fail("image dimensions overflow");
    }

    const size_t pixels = static_cast<size_t>(info.width) * static_cast<size_t>(info.height);
    if (pixels > limits.max_image_pixels) {
        return Fail("image pixel count exceeds limit");
    }

    return Ok();
}

ValidationResult ValidateTokenIds(const google::protobuf::RepeatedField<int64_t>& values, int64_t max_token_id) {
    for (int64_t value : values) {
        if (value < 0 || value > max_token_id) {
            return Fail("input_ids contains out-of-range token id");
        }
    }
    return Ok();
}

ValidationResult ValidateAttentionMask(const google::protobuf::RepeatedField<int64_t>& values) {
    for (int64_t value : values) {
        if (value != 0 && value != 1) {
            return Fail("attention_mask must contain only 0 or 1");
        }
    }
    return Ok();
}

ValidationResult ValidatePersonality(const google::protobuf::RepeatedField<float>& values, float max_abs) {
    for (float value : values) {
        if (!std::isfinite(value) || std::fabs(value) > max_abs) {
            return Fail("personality contains invalid value");
        }
    }
    return Ok();
}

} 

ValidationResult Ok() {
    return {};
}

ValidationResult Fail(std::string error) {
    return ValidationResult{false, std::move(error)};
}

ValidationResult ValidateAuth(
    const grpc::ServerContext& context,
    const AuthOptions& options) {
    std::vector<std::pair<std::string, std::string>> metadata;
    metadata.reserve(context.client_metadata().size());
    for (const auto& [key, value] : context.client_metadata()) {
        metadata.emplace_back(
            std::string(key.data(), key.length()),
            std::string(value.data(), value.length()));
    }
    return ValidateAuthMetadata(metadata, options);
}

ValidationResult ValidateAuthMetadata(
    const std::vector<std::pair<std::string, std::string>>& metadata,
    const AuthOptions& options) {
    if (options.token.empty()) {
        return Ok();
    }
    if (options.metadata_key.empty()) {
        return Fail("auth metadata key is empty");
    }

    const std::string key = ToLowerAscii(options.metadata_key);
    for (const auto& [metadata_key, value] : metadata) {
        if (ToLowerAscii(metadata_key) == key && ConstantTimeEquals(value, options.token)) {
            return Ok();
        }
    }

    return Fail("invalid auth token");
}

ValidationResult ValidateEmotionRequest(
    const multimodal_inference::EmotionRequest& request,
    const RequestLimits& limits) {
    if (request.input_ids_size() <= 0 || request.attention_mask_size() <= 0) {
        return Fail("Empty input");
    }
    if (request.input_ids_size() > limits.max_sequence_length) {
        return Fail("input_ids exceeds sequence length limit");
    }
    if (request.input_ids_size() != request.attention_mask_size()) {
        return Fail("input_ids and attention_mask size mismatch");
    }
    if (request.personality_size() != 11) {
        return Fail("personality must have 11 elements");
    }
    if (auto result = ValidateTokenIds(request.input_ids(), limits.max_token_id); !result.ok) {
        return result;
    }
    if (auto result = ValidateAttentionMask(request.attention_mask()); !result.ok) {
        return result;
    }
    if (auto result = ValidatePersonality(request.personality(), limits.max_abs_personality); !result.ok) {
        return result;
    }
    return Ok();
}

ValidationResult ValidateEmotionBatchRequest(
    const multimodal_inference::EmotionBatchRequest& request,
    const RequestLimits& limits) {
    if (request.batch_size() <= 0) {
        return Fail("Empty batch");
    }
    if (request.seq_length() <= 0) {
        return Fail("seq_length must be positive");
    }
    if (request.batch_size() > limits.max_batch_size) {
        return Fail("batch_size exceeds limit");
    }
    if (request.seq_length() > limits.max_sequence_length) {
        return Fail("seq_length exceeds limit");
    }

    const size_t batch_size = static_cast<size_t>(request.batch_size());
    const size_t seq_len = static_cast<size_t>(request.seq_length());
    if (batch_size > std::numeric_limits<size_t>::max() / seq_len) {
        return Fail("batch dimensions overflow");
    }

    const size_t expected_input_size = batch_size * seq_len;
    const size_t expected_personality_size = batch_size * 11;
    if (static_cast<size_t>(request.input_ids_size()) != expected_input_size ||
        static_cast<size_t>(request.attention_mask_size()) != expected_input_size ||
        static_cast<size_t>(request.personality_size()) != expected_personality_size) {
        return Fail("Input size mismatch");
    }
    if (auto result = ValidateTokenIds(request.input_ids(), limits.max_token_id); !result.ok) {
        return result;
    }
    if (auto result = ValidateAttentionMask(request.attention_mask()); !result.ok) {
        return result;
    }
    if (auto result = ValidatePersonality(request.personality(), limits.max_abs_personality); !result.ok) {
        return result;
    }
    return Ok();
}

ValidationResult ValidateVLMRequest(
    const multimodal_inference::VLMRequest& request,
    const RequestLimits& limits) {
    if (auto result = ValidateText(request.prompt(), limits.max_prompt_bytes, true, "prompt"); !result.ok) {
        return result;
    }
    if (auto result = ValidateMetadataText(request.session_id(), limits.max_vlm_session_id_bytes, "session_id"); !result.ok) {
        return result;
    }
    if (auto result = ValidateMetadataText(request.request_id(), limits.max_vlm_request_id_bytes, "request_id"); !result.ok) {
        return result;
    }
    if (auto result = ValidateMetadataText(request.task_type(), limits.max_vlm_task_type_bytes, "task_type"); !result.ok) {
        return result;
    }

    if (request.image_source_case() == multimodal_inference::VLMRequest::kImagePath) {
        return Fail("image_path is disabled");
    }
    if (request.image_source_case() == multimodal_inference::VLMRequest::kImageData) {
        if (auto result = ValidateImageData(request.image_data(), limits); !result.ok) {
            return result;
        }
    }

    if (!std::isfinite(request.temperature()) || request.temperature() < 0.0f ||
        request.temperature() > limits.max_temperature) {
        return Fail("temperature out of range");
    }
    if (!std::isfinite(request.top_p()) || request.top_p() < 0.0f || request.top_p() > 1.0f) {
        return Fail("top_p out of range");
    }
    if (request.top_k() < 0 || request.top_k() > limits.max_top_k) {
        return Fail("top_k out of range");
    }
    if (request.max_tokens() < 0 || request.max_tokens() > limits.max_vlm_tokens) {
        return Fail("max_tokens out of range");
    }
    if (request.context_size() < 0 || request.context_size() > limits.max_context_size) {
        return Fail("context_size out of range");
    }
    if (request.context_size() > 0 && request.context_size() < limits.min_context_size) {
        return Fail("context_size below minimum");
    }

    return Ok();
}

ValidationResult ValidateSaliencyRequest(
    const multimodal_inference::SaliencyRequest& request,
    const RequestLimits& limits) {
    if (request.timestamp_ms() < 0) {
        return Fail("timestamp_ms must be non-negative");
    }
    if (request.width() < 0 || request.height() < 0) {
        return Fail("frame dimensions must be non-negative");
    }
    if (request.width() > 0 && static_cast<uint32_t>(request.width()) > limits.max_image_width) {
        return Fail("frame width exceeds limit");
    }
    if (request.height() > 0 && static_cast<uint32_t>(request.height()) > limits.max_image_height) {
        return Fail("frame height exceeds limit");
    }
    return ValidateImageData(request.frame_data(), limits);
}

}
