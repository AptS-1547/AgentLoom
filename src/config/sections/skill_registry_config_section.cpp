#include "config_section.h"
#include "../../skill/skill_manifest_json.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <unordered_set>

namespace server_config {
namespace {

using Json = nlohmann::json;

std::filesystem::path ResolveRelativeToConfig(
    const std::filesystem::path& path,
    const std::filesystem::path& config_path) {
    if (path.empty() || path.is_absolute() || config_path.empty()) return path;
    return config_path.parent_path() / path;
}

Json ReadJsonFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("failed to open Skill manifest: " + path.string());
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF) {
        text.erase(0, 3);
    }
    try {
        return Json::parse(text);
    } catch (const Json::exception& error) {
        throw std::runtime_error("invalid Skill manifest " + path.string() + ": " + error.what());
    }
}

agent::skill::SkillManifest ParseManifest(const Json& item, std::string_view source) {
    if (!item.is_object()) {
        throw std::runtime_error(std::string(source) + " must contain a JSON object manifest");
    }
    return agent::skill::ParseSkillManifestJson(item);
}

void AppendManifestJson(const Json& value,
                        std::string_view source,
                        std::vector<agent::skill::SkillManifest>& output) {
    if (value.is_array()) {
        for (const auto& item : value) output.push_back(ParseManifest(item, source));
        return;
    }
    if (value.is_object() && value.contains("manifests")) {
        if (!value["manifests"].is_array()) throw std::runtime_error(std::string(source) + ".manifests must be an array");
        for (const auto& item : value["manifests"]) output.push_back(ParseManifest(item, source));
        return;
    }
    output.push_back(ParseManifest(value, source));
}

DECLARE_CONFIG_SECTION(SkillRegistryConfigSection, "skills")
    void Validate(MultimodalServerOptions& options) const override;
};

bool SkillRegistryConfigSection::LoadCli(CliCursor&, MultimodalServerOptions&) const {
    return false;
}

void SkillRegistryConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (!section) return;
    SetBool(*section, Name(), "enabled", options.skills.enabled);
    SetPath(*section, Name(), "manifest_directory", options.skills.manifest_directory);
    if (const Json* legacy_directory = FindField(*section, Name(), "directory")) {
        if (!legacy_directory->is_string()) throw std::runtime_error("skills.directory must be a string");
        options.skills.manifest_directory = legacy_directory->get<std::string>();
    }
    SetString(*section, Name(), "manifest_filename_regex", options.skills.manifest_filename_regex);
    const Json* manifests = FindField(*section, Name(), "manifests");
    if (manifests) AppendManifestJson(*manifests, "skills.manifests", options.skill_manifests);
}

void SkillRegistryConfigSection::Validate(MultimodalServerOptions& options) const {
    options.skills.manifest_directory = ResolveRelativeToConfig(
        options.skills.manifest_directory, options.config_file_path);
    if (!options.skills.enabled) return;
    try {
        const std::regex filename_regex(options.skills.manifest_filename_regex);
        if (!options.skills.manifest_directory.empty()) {
            if (!std::filesystem::exists(options.skills.manifest_directory) ||
                !std::filesystem::is_directory(options.skills.manifest_directory)) {
                throw std::runtime_error("skills.manifest_directory is not a directory: " + options.skills.manifest_directory.string());
            }
            std::vector<std::filesystem::path> files;
            for (const auto& entry : std::filesystem::directory_iterator(options.skills.manifest_directory)) {
                if (entry.is_regular_file() && std::regex_match(entry.path().filename().string(), filename_regex)) {
                    files.push_back(entry.path());
                }
            }
            std::sort(files.begin(), files.end());
            for (const auto& file : files) AppendManifestJson(ReadJsonFile(file), file.string(), options.skill_manifests);
        }
    } catch (const std::regex_error& error) {
        throw std::runtime_error(std::string("invalid skills.manifest_filename_regex: ") + error.what());
    }
    std::unordered_set<std::string> seen_ids;
    std::unordered_set<std::string> seen_tools;
    for (const auto& manifest : options.skill_manifests) {
        if (manifest.skill_id.empty() || manifest.version.empty()) {
            throw std::runtime_error("skills.manifests requires skill_id and version");
        }
        const auto tool_name = manifest.tool_name.empty() ? manifest.skill_id : manifest.tool_name;
        if (!seen_ids.insert(manifest.skill_id + "@" + manifest.version).second) {
            throw std::runtime_error("duplicate Skill manifest: " + manifest.skill_id + "@" + manifest.version);
        }
        if (!seen_tools.insert(tool_name).second) {
            throw std::runtime_error("duplicate Skill tool_name: " + tool_name);
        }
        try {
            if (!nlohmann::json::parse(manifest.input_schema_json).is_object() ||
                !nlohmann::json::parse(manifest.output_schema_json).is_object()) {
                throw std::runtime_error("skills.manifests schemas must be objects");
            }
        } catch (const nlohmann::json::exception& error) {
            throw std::runtime_error(std::string("invalid skill schema: ") + error.what());
        }
    }
}

} // namespace
REGISTER_CONFIG_SECTION(SkillRegistryConfigSection)
} // namespace server_config
