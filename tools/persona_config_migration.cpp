#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

int Fail(const std::string& message) {
    std::cerr << "[persona-migrate] FAIL: " << message << '\n';
    return 1;
}

Json LoadJson(const fs::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path.string());
    }
    return Json::parse(file);
}

std::string PersonaIdFromPath(const fs::path& path) {
    auto stem = path.stem().string();
    constexpr std::string_view prefix = "config_";
    if (stem.starts_with(prefix)) {
        stem.erase(0, prefix.size());
    }
    return stem;
}

Json MigrateOne(const fs::path& path) {
    auto source = LoadJson(path);
    const auto personality = source.value("personality", Json::object());
    const auto agent = source.value("agent", Json::object());
    const auto memory = source.value("memory", Json::object());

    Json out;
    out["personaId"] = PersonaIdFromPath(path);
    out["sourceFile"] = path.filename().string();
    out["displayName"] = personality.value("name", out["personaId"].get<std::string>());
    out["defaultPersona"] = false;
    out["proactiveLevel"] = agent.value("proactive_level", "off");
    out["contextIds"] = Json::array();
    out["contextPatterns"] = Json::array();

    Json p;
    p["name"] = personality.value("name", out["personaId"].get<std::string>());
    p["description"] = personality.value("description", "");
    p["traits"] = personality.value("traits", std::vector<std::string>{});
    p["openness"] = personality.value("openness", 0.5);
    p["extraversion"] = personality.value("extraversion", 0.5);
    p["humorTendency"] = personality.value("humor_tendency", 0.5);
    p["empathyLevel"] = personality.value("empathy_level", 0.5);
    p["curiosityLevel"] = personality.value("curiosity_level", 0.5);
    p["formality"] = personality.value("formality", 0.5);
    p["conscientiousness"] = personality.value("conscientiousness", 0.5);
    p["agreeableness"] = personality.value("agreeableness", 0.5);
    p["neuroticism"] = personality.value("neuroticism", 0.5);
    p["verbosity"] = personality.value("verbosity", 0.5);
    p["emotionalBaseline"] = personality.value("emotional_baseline", "");
    p["emotionalVolatility"] = personality.value("emotional_volatility", 0.5);
    out["personality"] = std::move(p);

    out["memory"] = {
        {"sourceUserId", memory.value("user_id", "")},
        {"sourceCollectionName", memory.value("collection_name", "")},
        {"l3SearchLimit", memory.value("l3_search_limit", 5)},
        {"relevanceThreshold", memory.value("relevance_threshold", 0.3)}
    };

    if (source.contains("emotion_prompts")) {
        out["emotionPrompts"] = source["emotion_prompts"];
    }
    if (source.contains("emotion_state")) {
        out["emotionState"] = source["emotion_state"];
    }
    if (source.contains("attention")) {
        out["attention"] = source["attention"];
    }

    return out;
}

std::vector<fs::path> CollectInputs(const fs::path& input) {
    std::vector<fs::path> files;
    if (fs::is_regular_file(input)) {
        files.push_back(input);
        return files;
    }
    for (const auto& entry : fs::directory_iterator(input)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto path = entry.path();
        const auto name = path.filename().string();
        if (path.extension() == ".json" && name.starts_with("config_")) {
            files.push_back(path);
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <python-config-file-or-dir> <output.json>\n";
        return 2;
    }

    try {
        const fs::path input = argv[1];
        const fs::path output = argv[2];
        auto files = CollectInputs(input);
        if (files.empty()) {
            return Fail("no config_*.json files found");
        }

        Json registry;
        registry["schemaVersion"] = "persona_gateway.persona_registry.v1";
        registry["source"] = fs::absolute(input).string();
        registry["personas"] = Json::array();
        for (const auto& file : files) {
            registry["personas"].push_back(MigrateOne(file));
        }
        if (!registry["personas"].empty()) {
            registry["personas"][0]["defaultPersona"] = true;
        }

        fs::create_directories(output.parent_path());
        std::ofstream out(output, std::ios::binary | std::ios::trunc);
        if (!out) {
            return Fail("cannot open output: " + output.string());
        }
        out << registry.dump(2);
        out << '\n';
        std::cout << "[persona-migrate] wrote " << registry["personas"].size()
                  << " personas to " << output.string() << '\n';
        return 0;
    } catch (const std::exception& e) {
        return Fail(e.what());
    }
}
