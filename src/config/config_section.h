#pragma once

#include "server_options.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace server_config {

using Json = nlohmann::json;

inline constexpr std::string_view kConfigPathFlag = "--config";

struct CliCursor {
    int argc = 0;
    char** argv = nullptr;
    int& index;
};

class CliArgumentParser {
public:
    explicit CliArgumentParser(CliCursor& cursor);

    std::optional<std::string> Value(std::string_view flag);
    bool Flag(std::string_view flag) const;

private:
    CliCursor& cursor_;
    std::string_view arg_;
};

class IConfigSection {
public:
    virtual ~IConfigSection() = default;
    virtual std::string_view Name() const = 0;
    virtual void LoadJson(const Json& root, MultimodalServerOptions& options) const = 0;
    virtual bool LoadCli(CliCursor& cursor, MultimodalServerOptions& options) const = 0;
    virtual void Validate(MultimodalServerOptions& options) const;
};

using ConfigSectionFactory = std::unique_ptr<IConfigSection> (*)();

class ConfigSectionRegistry {
public:
    static ConfigSectionRegistry& Instance();

    bool Register(std::string_view name, ConfigSectionFactory factory);
    std::vector<std::unique_ptr<IConfigSection>> CreateSections() const;

private:
    struct Entry {
        std::string_view name;
        ConfigSectionFactory factory = nullptr;
    };

    std::vector<Entry> entries_;
};

class ConfigSectionSelection {
public:
    static ConfigSectionSelection All();
    static ConfigSectionSelection Only(std::initializer_list<std::string_view> names);
    static ConfigSectionSelection Only(std::vector<std::string> names);

    bool Includes(std::string_view name) const noexcept;

private:
    ConfigSectionSelection(bool include_all, std::vector<std::string> names);

    bool include_all_ = true;
    std::vector<std::string> names_;
};

template <typename T>
class ConfigSectionRegistrar {
public:
    ConfigSectionRegistrar() {
        ConfigSectionRegistry::Instance().Register(
            T::kName,
            []() -> std::unique_ptr<IConfigSection> {
                return std::make_unique<T>();
            });
    }
};

std::vector<std::unique_ptr<IConfigSection>> BuildConfigSections();
std::vector<std::unique_ptr<IConfigSection>> BuildConfigSections(const ConfigSectionSelection& selection);
void ValidateOptions(MultimodalServerOptions& options);
void ValidateOptions(MultimodalServerOptions& options, const ConfigSectionSelection& selection);

int ParseCliInt(std::string_view flag, const std::string& value);
int ParsePositiveOption(std::string_view flag, const std::string& value);
int ParseNonNegativeOption(std::string_view flag, const std::string& value);
std::size_t ParseMegabytesOption(std::string_view flag, const std::string& value);
std::size_t ParseOptionalMegabytesOption(std::string_view flag, const std::string& value);
float ParseFloatOption(std::string_view flag, const std::string& value, float minimum, float maximum);

const Json* FindSection(const Json& root, std::string_view name);
const Json* FindField(const Json& section, std::string_view section_name, std::string_view field_name);

void SetString(const Json& section, std::string_view section_name, std::string_view field_name, std::string& target);
void SetPath(const Json& section, std::string_view section_name, std::string_view field_name, std::filesystem::path& target);
void SetInt(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    int& target,
    int min_value = std::numeric_limits<int>::min(),
    int max_value = std::numeric_limits<int>::max());
void SetInt32(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    int32_t& target,
    int32_t min_value = std::numeric_limits<int32_t>::min(),
    int32_t max_value = std::numeric_limits<int32_t>::max());
void SetInt64(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    int64_t& target,
    int64_t min_value = std::numeric_limits<int64_t>::min(),
    int64_t max_value = std::numeric_limits<int64_t>::max());
void SetSize(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    size_t& target,
    size_t min_value = 0,
    size_t max_value = std::numeric_limits<size_t>::max());
void SetUInt32(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    uint32_t& target,
    uint32_t min_value = 0,
    uint32_t max_value = std::numeric_limits<uint32_t>::max());
void SetFloat(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    float& target,
    float min_value,
    float max_value);
void SetBool(const Json& section, std::string_view section_name, std::string_view field_name, bool& target);
void SetMegabytes(
    const Json& section,
    std::string_view section_name,
    std::string_view field_name,
    size_t& target,
    int64_t min_mb = 0);

} // namespace server_config

#define DECLARE_CONFIG_SECTION(ClassName, SectionNameLiteral) \
class ClassName final : public ::server_config::IConfigSection { \
public: \
    static constexpr std::string_view kName = SectionNameLiteral; \
    std::string_view Name() const override { return kName; } \
    void LoadJson(const ::server_config::Json& root, MultimodalServerOptions& options) const override; \
    bool LoadCli(::server_config::CliCursor& cursor, MultimodalServerOptions& options) const override;

#define CONFIG_CLI_STRING(Name, Literal) \
    static constexpr std::string_view Name = Literal

#define CONFIG_VALUE_ARG(FlagName, ValueName, Body) \
    if (auto ValueName = parser.Value(FlagName)) { \
        Body \
        return true; \
    }

#define CONFIG_FLAG_ARG(FlagName, Body) \
    if (parser.Flag(FlagName)) { \
        Body \
        return true; \
    }

#define REGISTER_CONFIG_SECTION(ClassName) \
static const ::server_config::ConfigSectionRegistrar<ClassName> g_##ClassName##_registrar;
