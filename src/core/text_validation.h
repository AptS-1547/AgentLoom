#pragma once

#include <cstdint>
#include <string_view>

namespace core {

// 严格校验 UTF-8，拒绝过长编码、代理项和超出 Unicode 范围的码点。
inline bool IsValidUtf8(std::string_view value) noexcept {
    std::size_t index = 0;
    while (index < value.size()) {
        const auto lead = static_cast<unsigned char>(value[index]);
        if (lead <= 0x7F) {
            ++index;
            continue;
        }

        std::uint32_t code_point = 0;
        std::size_t continuation_count = 0;
        std::uint32_t minimum = 0;
        if ((lead & 0xE0) == 0xC0) {
            code_point = lead & 0x1F;
            continuation_count = 1;
            minimum = 0x80;
        } else if ((lead & 0xF0) == 0xE0) {
            code_point = lead & 0x0F;
            continuation_count = 2;
            minimum = 0x800;
        } else if ((lead & 0xF8) == 0xF0) {
            code_point = lead & 0x07;
            continuation_count = 3;
            minimum = 0x10000;
        } else {
            return false;
        }
        if (index + continuation_count >= value.size()) {
            return false;
        }
        for (std::size_t offset = 1; offset <= continuation_count; ++offset) {
            const auto next = static_cast<unsigned char>(value[index + offset]);
            if ((next & 0xC0) != 0x80) {
                return false;
            }
            code_point = (code_point << 6) | (next & 0x3F);
        }
        if (code_point < minimum || code_point > 0x10FFFF ||
            (code_point >= 0xD800 && code_point <= 0xDFFF)) {
            return false;
        }
        index += continuation_count + 1;
    }
    return true;
}

inline bool HasInvalidTextControl(std::string_view value) noexcept {
    for (const unsigned char ch : value) {
        if (ch == 0 || (ch < 0x20 && ch != '\n' && ch != '\r' && ch != '\t')) {
            return true;
        }
    }
    return false;
}

inline bool IsBlankAscii(std::string_view value) noexcept {
    if (value.empty()) {
        return true;
    }
    for (const unsigned char ch : value) {
        if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
            return false;
        }
    }
    return true;
}

}
