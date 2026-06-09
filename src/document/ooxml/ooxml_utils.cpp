#include "document_types.h"

#include <algorithm>
#include <cctype>
#include <regex>

namespace agent::document {
namespace {

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool RegexSearch(std::string_view text, const char* pattern) {
    return std::regex_search(std::string(text), std::regex(pattern, std::regex::icase));
}

bool RegexMatch(std::string_view text, const char* pattern) {
    return std::regex_match(std::string(text), std::regex(pattern, std::regex::icase));
}

std::size_t CountSubstr(std::string_view text, std::string_view needle) {
    if (needle.empty()) {
        return 0;
    }
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string_view::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

int ChineseOrdinalIndex(std::string_view token) {
    static constexpr std::string_view numerals[] = {
        "一", "二", "三", "四", "五", "六", "七", "八", "九"
    };
    for (int i = 0; i < 9; ++i) {
        if (token == numerals[i]) {
            return i + 1;
        }
    }
    return 0;
}

int ChineseOrdinalSuffixIndex(std::string_view text) {
    static constexpr std::string_view numerals[] = {
        "一", "二", "三", "四", "五", "六", "七", "八", "九"
    };
    for (int i = 0; i < 9; ++i) {
        if (text.ends_with(numerals[i])) {
            return i + 1;
        }
    }
    return 0;
}

} // namespace

std::string CleanText(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool in_space = false;
    for (unsigned char ch : text) {
        const bool space = std::isspace(ch) != 0;
        if (space) {
            in_space = true;
            continue;
        }
        if (in_space && !out.empty()) {
            out.push_back(' ');
        }
        out.push_back(static_cast<char>(ch));
        in_space = false;
    }
    return out;
}

std::optional<int> SafeInt(std::string_view text) {
    if (text.empty()) {
        return std::nullopt;
    }
    try {
        std::size_t consumed = 0;
        const auto value = std::stoi(std::string(text), &consumed);
        if (consumed == text.size()) {
            return value;
        }
    } catch (...) {
    }
    return std::nullopt;
}

std::optional<int> DetectNumberingLevel(std::string_view text) {
    const auto clean = CleanText(text);
    if (clean.empty()) {
        return std::nullopt;
    }
    if (RegexSearch(clean, R"(^第([一二三四五六七八九十百0-9]+)[章节])")) {
        return 1;
    }
    if (RegexSearch(clean, R"(^[一二三四五六七八九十]+[、.．])")) {
        return 1;
    }
    if (RegexSearch(clean, R"(^（[一二三四五六七八九十]+）)")) {
        return 2;
    }

    std::smatch match;
    const std::regex dotted(R"(^(\d+(?:[.．]\d+){0,4})[.．、\s])");
    if (std::regex_search(clean, match, dotted) && match.size() > 1) {
        const auto numbering = match[1].str();
        const auto dots = std::count(numbering.begin(), numbering.end(), '.')
            + CountSubstr(numbering, "．");
        return std::min(5, static_cast<int>(dots) + 1);
    }
    if (RegexSearch(clean, R"(^[A-Z][.．、\s])")) {
        return 2;
    }
    return std::nullopt;
}

std::optional<int> HeadingLevelFromStyle(std::string_view style) {
    const auto raw = CleanText(style);
    if (raw.empty()) {
        return std::nullopt;
    }
    std::smatch match;
    if (std::regex_search(raw, match, std::regex(R"(heading\s*([1-9]))", std::regex::icase))) {
        return std::stoi(match[1].str());
    }
    if (std::regex_search(raw, match, std::regex(R"(标题\s*([1-9一二三四五六七八九]))"))) {
        const auto token = match[1].str();
        if (token.size() == 1 && std::isdigit(static_cast<unsigned char>(token[0]))) {
            return token[0] - '0';
        }
        const auto index = ChineseOrdinalIndex(token);
        if (index > 0) {
            return index;
        }
    }
    if (raw.rfind("标题", 0) == 0) {
        const auto index = ChineseOrdinalSuffixIndex(raw);
        if (index > 0) {
            return index;
        }
    }
    const auto lowered = ToLowerAscii(raw);
    if (lowered == "title" || raw == "标题") {
        return 1;
    }
    if (lowered == "subtitle" || raw == "副标题") {
        return 2;
    }
    return std::nullopt;
}

std::optional<int> InferHeadingLevel(const DocumentBlock& block) {
    if (block.heading_level) {
        return block.heading_level;
    }
    if (auto style = HeadingLevelFromStyle(block.style)) {
        return style;
    }
    if (auto numbering = DetectNumberingLevel(block.text)) {
        return numbering;
    }
    if (block.slide && block.kind == "heading") {
        return 1;
    }
    if (block.font_size && *block.font_size >= 18.0 && block.text.size() <= 40) {
        return 2;
    }
    if (block.bold && block.text.size() <= 32) {
        return 3;
    }
    return std::nullopt;
}

std::string DocumentBlockId(std::string_view prefix, std::uint64_t order) {
    return std::string(prefix) + "-" + std::to_string(order);
}

} // namespace agent::document
