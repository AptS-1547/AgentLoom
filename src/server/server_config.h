#pragma once

#include "server_options.h"

#include <filesystem>
#include <optional>

namespace server_config {

std::optional<std::filesystem::path> FindConfigPath(int argc, char** argv);
void LoadConfigFile(const std::filesystem::path& path, MultimodalServerOptions& options);

}
