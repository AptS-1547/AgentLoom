#include "config_section.h"

#include "server_common.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace server_config {

void IConfigSection::Validate(MultimodalServerOptions&) const {}

CliArgumentParser::CliArgumentParser(CliCursor& cursor)
    : cursor_(cursor),
      arg_(cursor.argv[cursor.index]) {}

std::optional<std::string> CliArgumentParser::Value(std::string_view flag) {
    if (arg_ == flag) {
        if (cursor_.index + 1 >= cursor_.argc) {
            throw std::runtime_error(std::string(flag) + " requires a value");
        }
        return cursor_.argv[++cursor_.index];
    }
    if (arg_.size() > flag.size() &&
        arg_.starts_with(flag) &&
        arg_[flag.size()] == '=') {
        return std::string(arg_.substr(flag.size() + 1));
    }
    return std::nullopt;
}

bool CliArgumentParser::Flag(std::string_view flag) const {
    return arg_ == flag;
}

ConfigSectionRegistry& ConfigSectionRegistry::Instance() {
    static ConfigSectionRegistry registry;
    return registry;
}

bool ConfigSectionRegistry::Register(std::string_view name, ConfigSectionFactory factory) {
    auto duplicate = std::find_if(
        entries_.begin(),
        entries_.end(),
        [name](const Entry& entry) {
            return entry.name == name;
        });
    if (duplicate == entries_.end()) {
        entries_.push_back({name, factory});
    }
    return true;
}

std::vector<std::unique_ptr<IConfigSection>> ConfigSectionRegistry::CreateSections() const {
    std::vector<std::unique_ptr<IConfigSection>> sections;
    sections.reserve(entries_.size());
    for (const Entry& entry : entries_) {
        sections.push_back(entry.factory());
    }
    return sections;
}

std::vector<std::unique_ptr<IConfigSection>> BuildConfigSections() {
    return ConfigSectionRegistry::Instance().CreateSections();
}

void ValidateOptions(MultimodalServerOptions& options) {
    for (const auto& section : BuildConfigSections()) {
        section->Validate(options);
    }
}

int ParseCliInt(std::string_view flag, const std::string& value) {
    return server_common::ParseIntValue(std::string(flag), value.c_str());
}

int ParsePositiveOption(std::string_view flag, const std::string& value) {
    const int parsed = ParseCliInt(flag, value);
    if (parsed <= 0) {
        throw std::runtime_error(std::string(flag) + " must be positive");
    }
    return parsed;
}

int ParseNonNegativeOption(std::string_view flag, const std::string& value) {
    const int parsed = ParseCliInt(flag, value);
    if (parsed < 0) {
        throw std::runtime_error(std::string(flag) + " must be non-negative");
    }
    return parsed;
}

std::size_t ParseMegabytesOption(std::string_view flag, const std::string& value) {
    return static_cast<std::size_t>(ParsePositiveOption(flag, value)) * 1024 * 1024;
}

std::size_t ParseOptionalMegabytesOption(std::string_view flag, const std::string& value) {
    return static_cast<std::size_t>(ParseNonNegativeOption(flag, value)) * 1024 * 1024;
}

const Json* FindSection(const Json& root, std::string_view name) {
    auto it = root.find(std::string(name));
    if (it == root.end()) {
        return nullptr;
    }
    if (!it->is_object()) {
        throw std::runtime_error(std::string(name) + " must be an object");
    }
    return &*it;
}

const Json* FindField(const Json& section, std::string_view section_name, std::string_view field_name) {
    auto it = section.find(std::string(field_name));
    if (it == section.end()) {
        return nullptr;
    }
    if (it->is_null()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must not be null");
    }
    return &*it;
}

void SetString(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    std::string& target) {
    const Json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    if (!value->is_string()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be a string");
    }
    target = value->get<std::string>();
}

void SetPath(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    std::filesystem::path& target) {
    const Json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    if (!value->is_string()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be a string");
    }
    target = value->get<std::string>();
}

int64_t ReadInteger(
    const Json& value,
    std::string_view section_name,
    std::string_view field_name,
    int64_t min_value,
    int64_t max_value) {
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be an integer");
    }
    int64_t parsed = 0;
    try {
        parsed = value.get<int64_t>();
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " is out of range");
    }
    if (parsed < min_value || parsed > max_value) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " is out of range");
    }
    return parsed;
}

void SetInt(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    int& target,
    int min_value,
    int max_value) {
    const Json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    target = static_cast<int>(ReadInteger(*value, section_name, field_name, min_value, max_value));
}

void SetInt32(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    int32_t& target,
    int32_t min_value,
    int32_t max_value) {
    const Json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    target = static_cast<int32_t>(ReadInteger(*value, section_name, field_name, min_value, max_value));
}

void SetInt64(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    int64_t& target,
    int64_t min_value,
    int64_t max_value) {
    const Json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    target = ReadInteger(*value, section_name, field_name, min_value, max_value);
}

void SetSize(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    size_t& target,
    size_t min_value,
    size_t max_value) {
    const Json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    const int64_t parsed = ReadInteger(
        *value,
        section_name,
        field_name,
        static_cast<int64_t>(std::min<size_t>(min_value, static_cast<size_t>(std::numeric_limits<int64_t>::max()))),
        static_cast<int64_t>(std::min<size_t>(max_value, static_cast<size_t>(std::numeric_limits<int64_t>::max()))));
    target = static_cast<size_t>(parsed);
}

void SetUInt32(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    uint32_t& target,
    uint32_t min_value,
    uint32_t max_value) {
    const Json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    target = static_cast<uint32_t>(ReadInteger(*value, section_name, field_name, min_value, max_value));
}

void SetFloat(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    float& target,
    float min_value,
    float max_value) {
    const Json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    if (!value->is_number()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be a number");
    }
    const double parsed = value->get<double>();
    if (!std::isfinite(parsed) || parsed < min_value || parsed > max_value) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " is out of range");
    }
    target = static_cast<float>(parsed);
}

void SetBool(const Json& section, std::string_view section_name, std::string_view field_name, bool& target) {
    const Json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    if (!value->is_boolean()) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be a boolean");
    }
    target = value->get<bool>();
}

size_t MegabytesToBytes(int64_t mb, std::string_view section_name, std::string_view field_name) {
    if (mb < 0) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " must be non-negative");
    }
    const auto max_mb = static_cast<int64_t>(std::numeric_limits<size_t>::max() / (1024ULL * 1024ULL));
    if (mb > max_mb) {
        throw std::runtime_error(std::string(section_name) + "." + std::string(field_name) + " is out of range");
    }
    return static_cast<size_t>(mb) * 1024ULL * 1024ULL;
}

void SetMegabytes(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    size_t& target,
    int64_t min_mb) {
    const Json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    const int64_t parsed = ReadInteger(
        *value,
        section_name,
        field_name,
        min_mb,
        static_cast<int64_t>(std::numeric_limits<size_t>::max() / (1024ULL * 1024ULL)));
    target = MegabytesToBytes(parsed, section_name, field_name);
}

} // namespace server_config
