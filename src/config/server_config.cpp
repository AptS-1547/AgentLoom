#include "server_config.h"

#include "config_section.h"

#include <fstream>
#include <stdexcept>

namespace server_config {

std::optional<std::filesystem::path> FindConfigPath(int argc, char** argv) {
    std::optional<std::filesystem::path> path;
    for (int i = 1; i < argc; ++i) {
        CliCursor cursor{argc, argv, i};
        CliArgumentParser parser(cursor);
        if (auto value = parser.Value(kConfigPathFlag)) {
            path = *value;
        }
    }
    return path;
}

void LoadConfigFile(const std::filesystem::path& path, MultimodalServerOptions& options) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open config file: " + path.string());
    }

    Json root;
    try {
        file >> root;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse config file " + path.string() + ": " + e.what());
    }

    if (!root.is_object()) {
        throw std::runtime_error("Config file root must be an object");
    }

    for (const auto& section : BuildConfigSections()) {
        section->LoadJson(root, options);
    }
}

void ApplyCliFallbackOptions(int argc, char** argv, MultimodalServerOptions& options) {
    const auto sections = BuildConfigSections();
    for (int i = 1; i < argc; ++i) {
        CliCursor cursor{argc, argv, i};
        for (const auto& section : sections) {
            if (section->LoadCli(cursor, options)) {
                break;
            }
        }
    }
}

} // namespace server_config
