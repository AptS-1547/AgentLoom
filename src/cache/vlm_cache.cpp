#include "vlm_cache.h"

#include "../core/logger_adapter.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <utility>

namespace vlm_cache {
namespace {

static core::LoggerAdapter logger = core::LoggerAdapter::ForModule("cache");

constexpr uint32_t kRecordMagic = 0x434D4C56;
constexpr uint32_t kRecordVersion = 1;

int64_t NowMs() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

std::string HexBytes(const uint8_t* data, size_t size) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.resize(size * 2);
    for (size_t i = 0; i < size; ++i) {
        out[i * 2] = digits[data[i] >> 4];
        out[i * 2 + 1] = digits[data[i] & 0x0F];
    }
    return out;
}

uint32_t ReadBe32(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) |
           (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) |
           static_cast<uint32_t>(data[3]);
}

class Sha256State {
public:
    void Update(const void* data, size_t size) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        total_bytes_ += size;

        while (size > 0) {
            const size_t take = std::min(size, block_.size() - block_used_);
            std::memcpy(block_.data() + block_used_, bytes, take);
            block_used_ += take;
            bytes += take;
            size -= take;

            if (block_used_ == block_.size()) {
                ProcessBlock(block_.data());
                block_used_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> Final() {
        const uint64_t bit_length = static_cast<uint64_t>(total_bytes_) * 8ULL;

        block_[block_used_++] = 0x80;
        if (block_used_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_used_), block_.end(), 0);
            ProcessBlock(block_.data());
            block_used_ = 0;
        }

        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_used_), block_.begin() + 56, 0);
        for (int i = 0; i < 8; ++i) {
            block_[56 + i] = static_cast<uint8_t>((bit_length >> ((7 - i) * 8)) & 0xFF);
        }
        ProcessBlock(block_.data());

        std::array<uint8_t, 32> digest{};
        for (size_t i = 0; i < state_.size(); ++i) {
            digest[i * 4] = static_cast<uint8_t>((state_[i] >> 24) & 0xFF);
            digest[i * 4 + 1] = static_cast<uint8_t>((state_[i] >> 16) & 0xFF);
            digest[i * 4 + 2] = static_cast<uint8_t>((state_[i] >> 8) & 0xFF);
            digest[i * 4 + 3] = static_cast<uint8_t>(state_[i] & 0xFF);
        }
        return digest;
    }

private:
    static uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (~x & z);
    }

    static uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    static uint32_t BigSigma0(uint32_t x) {
        return std::rotr(x, 2) ^ std::rotr(x, 13) ^ std::rotr(x, 22);
    }

    static uint32_t BigSigma1(uint32_t x) {
        return std::rotr(x, 6) ^ std::rotr(x, 11) ^ std::rotr(x, 25);
    }

    static uint32_t SmallSigma0(uint32_t x) {
        return std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3);
    }

    static uint32_t SmallSigma1(uint32_t x) {
        return std::rotr(x, 17) ^ std::rotr(x, 19) ^ (x >> 10);
    }

    void ProcessBlock(const uint8_t* block) {
        static constexpr std::array<uint32_t, 64> k = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        };

        std::array<uint32_t, 64> w{};
        for (size_t i = 0; i < 16; ++i) {
            w[i] = ReadBe32(block + i * 4);
        }
        for (size_t i = 16; i < 64; ++i) {
            w[i] = SmallSigma1(w[i - 2]) + w[i - 7] + SmallSigma0(w[i - 15]) + w[i - 16];
        }

        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];

        for (size_t i = 0; i < 64; ++i) {
            const uint32_t t1 = h + BigSigma1(e) + Ch(e, f, g) + k[i] + w[i];
            const uint32_t t2 = BigSigma0(a) + Maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {
        0x6a09e667,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19
    };
    std::array<uint8_t, 64> block_{};
    size_t block_used_ = 0;
    size_t total_bytes_ = 0;
};

template <typename T>
void HashAppend(Sha256State& hash, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    hash.Update(&value, sizeof(value));
}

void HashAppendString(Sha256State& hash, std::string_view value) {
    const uint64_t size = static_cast<uint64_t>(value.size());
    HashAppend(hash, size);
    hash.Update(value.data(), value.size());
}

uint32_t FloatBits(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::filesystem::path ResultsDir(const Options& options) {
    return options.cache_dir / "results";
}

std::filesystem::path ImagesDir(const Options& options) {
    return options.cache_dir / "images";
}

std::filesystem::path PromptsDir(const Options& options) {
    return options.cache_dir / "prompts";
}

std::filesystem::path ResultPath(const Options& options, const std::string& cache_key) {
    return ResultsDir(options) / (cache_key + ".bin");
}

std::filesystem::path ImagePath(const Options& options, const std::string& image_sha256) {
    return ImagesDir(options) / (image_sha256 + ".bin");
}

std::filesystem::path PromptPath(const Options& options, const std::string& prompt_sha256) {
    return PromptsDir(options) / (prompt_sha256 + ".txt");
}

template <typename T>
void WriteValue(std::ostream& out, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) {
        throw std::runtime_error("write failed");
    }
}

void WriteString(std::ostream& out, std::string_view value) {
    const uint64_t size = static_cast<uint64_t>(value.size());
    WriteValue(out, size);
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!out) {
        throw std::runtime_error("write failed");
    }
}

template <typename T>
bool ReadValue(std::istream& in, T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return static_cast<bool>(in);
}

bool ReadString(std::istream& in, std::string& value, uint64_t max_size) {
    uint64_t size = 0;
    if (!ReadValue(in, size) || size > max_size) {
        return false;
    }
    value.resize(static_cast<size_t>(size));
    if (size == 0) {
        return true;
    }
    in.read(value.data(), static_cast<std::streamsize>(size));
    return static_cast<bool>(in);
}

std::string ReadSmallFile(const std::filesystem::path& path, size_t max_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    if (end < 0 || (max_size > 0 && static_cast<uint64_t>(end) > max_size)) {
        return {};
    }
    in.seekg(0, std::ios::beg);
    std::string value;
    value.resize(static_cast<size_t>(end));
    if (!value.empty()) {
        in.read(value.data(), static_cast<std::streamsize>(value.size()));
    }
    if (!in && !in.eof()) {
        return {};
    }
    return value;
}

std::vector<uint8_t> ReadByteFile(const std::filesystem::path& path, size_t max_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    if (end < 0 || (max_size > 0 && static_cast<uint64_t>(end) > max_size)) {
        return {};
    }
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> value(static_cast<size_t>(end));
    if (!value.empty()) {
        in.read(reinterpret_cast<char*>(value.data()), static_cast<std::streamsize>(value.size()));
    }
    if (!in && !in.eof()) {
        return {};
    }
    return value;
}

void WriteFileIfNeeded(const std::filesystem::path& path, const void* data, size_t size) {
    if (std::filesystem::exists(path)) {
        return;
    }
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("open failed");
    }
    if (size > 0) {
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    if (!out) {
        throw std::runtime_error("write failed");
    }
}

bool LooksLikeHash(std::string_view value) {
    if (value.size() != 64) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') ||
               (c >= 'a' && c <= 'f') ||
               (c >= 'A' && c <= 'F');
    });
}

}

std::string Sha256Hex(const void* data, size_t size) {
    Sha256State hash;
    hash.Update(data, size);
    const auto digest = hash.Final();
    return HexBytes(digest.data(), digest.size());
}

std::string Sha256Hex(std::string_view data) {
    return Sha256Hex(data.data(), data.size());
}

VLMCache::VLMCache(Options options)
    : options_(std::move(options)) {
    if (options_.enabled && options_.persist) {
        LoadPersisted();
    }
}

KeyInfo VLMCache::BuildKey(
    const std::vector<uint8_t>& image_data,
    std::string_view prompt,
    const llm::GenerateParams& params,
    std::string_view model_fingerprint) const {
    KeyInfo info;
    info.image_sha256 = image_data.empty()
        ? Sha256Hex("", 0)
        : Sha256Hex(image_data.data(), image_data.size());
    info.prompt_sha256 = Sha256Hex(prompt);

    Sha256State hash;
    HashAppendString(hash, "vlm-cache-v1");
    HashAppendString(hash, model_fingerprint);
    HashAppendString(hash, info.image_sha256);
    HashAppendString(hash, info.prompt_sha256);
    HashAppend(hash, params.max_tokens);
    HashAppend(hash, params.context_size);
    HashAppend(hash, params.top_k);
    const uint32_t temperature = FloatBits(params.temperature);
    const uint32_t top_p = FloatBits(params.top_p);
    HashAppend(hash, temperature);
    HashAppend(hash, top_p);
    const auto digest = hash.Final();
    info.cache_key = HexBytes(digest.data(), digest.size());
    return info;
}

std::optional<Result> VLMCache::Get(const std::string& cache_key) {
    if (!options_.enabled) {
        return std::nullopt;
    }

    std::lock_guard lock(mutex_);
    const int64_t now = NowMs();
    auto it = entries_.find(cache_key);
    if (it == entries_.end()) {
        return std::nullopt;
    }
    if (IsExpired(it->second, now)) {
        current_bytes_ -= std::min(current_bytes_, it->second.bytes);
        RemovePersistedResult(cache_key);
        entries_.erase(it);
        return std::nullopt;
    }
    return TouchAndMakeResult(it->second, now);
}

std::optional<Result> VLMCache::GetLatestFallback(
    std::string_view session_id,
    std::string_view task_type,
    std::string_view model_fingerprint) {
    if (!options_.enabled) {
        return std::nullopt;
    }

    std::lock_guard lock(mutex_);
    const int64_t now = NowMs();
    std::optional<std::string> best_key;
    int64_t best_time = std::numeric_limits<int64_t>::min();

    for (auto it = entries_.begin(); it != entries_.end();) {
        if (IsExpired(it->second, now)) {
            current_bytes_ -= std::min(current_bytes_, it->second.bytes);
            RemovePersistedResult(it->first);
            it = entries_.erase(it);
            continue;
        }

        const auto& record = it->second.record;
        if (!model_fingerprint.empty() && record.model_fingerprint != model_fingerprint) {
            ++it;
            continue;
        }
        if (!session_id.empty() && record.session_id != session_id) {
            ++it;
            continue;
        }
        if (!task_type.empty() && record.task_type != task_type) {
            ++it;
            continue;
        }

        const int64_t candidate_time = std::max(record.result.last_hit_at_ms, record.result.created_at_ms);
        if (!best_key || candidate_time > best_time) {
            best_key = it->first;
            best_time = candidate_time;
        }
        ++it;
    }

    if (!best_key) {
        return std::nullopt;
    }
    return TouchAndMakeResult(entries_.at(*best_key), now);
}

void VLMCache::Put(StoreRecord record) {
    if (!options_.enabled) {
        return;
    }
    if (record.key.cache_key.empty()) {
        return;
    }

    const int64_t now = NowMs();
    record.result.cache_key = record.key.cache_key;
    record.result.image_sha256 = record.key.image_sha256;
    record.result.prompt_sha256 = record.key.prompt_sha256;
    if (record.result.created_at_ms <= 0) {
        record.result.created_at_ms = now;
    }
    record.result.last_hit_at_ms = now;
    if (record.result.hit_count == 0) {
        record.result.hit_count = 1;
    }

    Entry entry;
    entry.bytes = EstimateBytes(record);
    entry.record = std::move(record);

    std::lock_guard lock(mutex_);
    auto old = entries_.find(entry.record.key.cache_key);
    if (old != entries_.end()) {
        current_bytes_ -= std::min(current_bytes_, old->second.bytes);
    }
    current_bytes_ += entry.bytes;

    const std::string key = entry.record.key.cache_key;
    if (options_.persist) {
        try {
            PersistEntry(entry);
        } catch (const std::exception& e) {
            logger.warn("[VLMCache] persist failed for {}: {}", key, e.what());
        }
    }
    entries_[key] = std::move(entry);
    EvictLocked();
}

const Options& VLMCache::options() const {
    return options_;
}

size_t VLMCache::size() const {
    std::lock_guard lock(mutex_);
    return entries_.size();
}

Result VLMCache::TouchAndMakeResult(Entry& entry, int64_t now_ms) {
    entry.record.result.last_hit_at_ms = now_ms;
    entry.record.result.hit_count += 1;
    if (options_.persist) {
        try {
            PersistEntry(entry);
        } catch (const std::exception& e) {
            logger.warn("[VLMCache] persist touch failed for {}: {}", entry.record.key.cache_key, e.what());
        }
    }
    return MakeResult(entry);
}

Result VLMCache::MakeResult(const Entry& entry) const {
    return entry.record.result;
}

size_t VLMCache::EstimateBytes(const StoreRecord& record) const {
    size_t bytes = record.result.text.size() +
        record.prompt.size() +
        record.image_data.size() +
        record.model_fingerprint.size() +
        record.session_id.size() +
        record.request_id.size() +
        record.task_type.size() +
        record.key.cache_key.size() +
        record.key.image_sha256.size() +
        record.key.prompt_sha256.size() +
        sizeof(record.params) +
        sizeof(record.result);
    return bytes;
}

bool VLMCache::IsExpired(const Entry& entry, int64_t now_ms) const {
    if (options_.ttl_seconds <= 0) {
        return false;
    }
    const int64_t ttl_ms = options_.ttl_seconds * 1000;
    return entry.record.result.created_at_ms > 0 &&
           now_ms - entry.record.result.created_at_ms > ttl_ms;
}

void VLMCache::EvictLocked() {
    const int64_t now = NowMs();
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (IsExpired(it->second, now)) {
            current_bytes_ -= std::min(current_bytes_, it->second.bytes);
            RemovePersistedResult(it->first);
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }

    while ((options_.max_entries > 0 && entries_.size() > options_.max_entries) ||
           (options_.max_bytes > 0 && current_bytes_ > options_.max_bytes)) {
        auto victim = entries_.end();
        int64_t victim_time = std::numeric_limits<int64_t>::max();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            const int64_t t = std::max(it->second.record.result.last_hit_at_ms, it->second.record.result.created_at_ms);
            if (t < victim_time) {
                victim = it;
                victim_time = t;
            }
        }
        if (victim == entries_.end()) {
            break;
        }
        current_bytes_ -= std::min(current_bytes_, victim->second.bytes);
        RemovePersistedResult(victim->first);
        entries_.erase(victim);
    }
}

void VLMCache::LoadPersisted() {
    try {
        std::filesystem::create_directories(ResultsDir(options_));
        std::filesystem::create_directories(ImagesDir(options_));
        std::filesystem::create_directories(PromptsDir(options_));
    } catch (const std::exception& e) {
        logger.warn("[VLMCache] create cache dirs failed: {}", e.what());
        return;
    }

    size_t loaded = 0;
    const auto dir = ResultsDir(options_);
    if (!std::filesystem::exists(dir)) {
        return;
    }

    std::lock_guard lock(mutex_);
    std::error_code iter_ec;
    for (const auto& item : std::filesystem::directory_iterator(dir, iter_ec)) {
        if (!item.is_regular_file() || item.path().extension() != ".bin") {
            continue;
        }
        auto entry = ReadEntry(item.path());
        if (!entry) {
            continue;
        }
        if (IsExpired(*entry, NowMs())) {
            RemovePersistedResult(entry->record.key.cache_key);
            continue;
        }
        current_bytes_ += entry->bytes;
        entries_[entry->record.key.cache_key] = std::move(*entry);
        ++loaded;
    }
    if (iter_ec) {
        logger.warn("[VLMCache] directory scan failed for {}: {}", dir.string(), iter_ec.message());
    }
    EvictLocked();
    logger.info("[VLMCache] loaded {} entries from {}", loaded, options_.cache_dir.string());
}

void VLMCache::PersistEntry(const Entry& entry) {
    std::filesystem::create_directories(ResultsDir(options_));
    if (options_.store_images && !entry.record.image_data.empty() && LooksLikeHash(entry.record.key.image_sha256)) {
        WriteFileIfNeeded(
            ImagePath(options_, entry.record.key.image_sha256),
            entry.record.image_data.data(),
            entry.record.image_data.size());
    }
    if (options_.store_prompts && !entry.record.prompt.empty() && LooksLikeHash(entry.record.key.prompt_sha256)) {
        WriteFileIfNeeded(
            PromptPath(options_, entry.record.key.prompt_sha256),
            entry.record.prompt.data(),
            entry.record.prompt.size());
    }

    const auto path = ResultPath(options_, entry.record.key.cache_key);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("open result failed");
    }

    WriteValue(out, kRecordMagic);
    WriteValue(out, kRecordVersion);
    WriteString(out, entry.record.key.cache_key);
    WriteString(out, entry.record.key.image_sha256);
    WriteString(out, entry.record.key.prompt_sha256);
    WriteString(out, entry.record.model_fingerprint);
    WriteString(out, entry.record.session_id);
    WriteString(out, entry.record.request_id);
    WriteString(out, entry.record.task_type);
    WriteValue(out, entry.record.params.max_tokens);
    WriteValue(out, entry.record.params.temperature);
    WriteValue(out, entry.record.params.context_size);
    WriteValue(out, entry.record.params.top_p);
    WriteValue(out, entry.record.params.top_k);
    WriteString(out, entry.record.result.text);
    WriteValue(out, entry.record.result.image_encode_ms);
    WriteValue(out, entry.record.result.prompt_eval_ms);
    WriteValue(out, entry.record.result.eval_ms);
    WriteValue(out, entry.record.result.prompt_tokens);
    WriteValue(out, entry.record.result.generated_tokens);
    WriteValue(out, entry.record.result.created_at_ms);
    WriteValue(out, entry.record.result.last_hit_at_ms);
    WriteValue(out, entry.record.result.hit_count);
}

std::optional<VLMCache::Entry> VLMCache::ReadEntry(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    if (!ReadValue(in, magic) || !ReadValue(in, version) ||
        magic != kRecordMagic || version != kRecordVersion) {
        return std::nullopt;
    }

    StoreRecord record;
    if (!ReadString(in, record.key.cache_key, 4096) ||
        !ReadString(in, record.key.image_sha256, 4096) ||
        !ReadString(in, record.key.prompt_sha256, 4096) ||
        !ReadString(in, record.model_fingerprint, 8192) ||
        !ReadString(in, record.session_id, 1024) ||
        !ReadString(in, record.request_id, 1024) ||
        !ReadString(in, record.task_type, 1024) ||
        !ReadValue(in, record.params.max_tokens) ||
        !ReadValue(in, record.params.temperature) ||
        !ReadValue(in, record.params.context_size) ||
        !ReadValue(in, record.params.top_p) ||
        !ReadValue(in, record.params.top_k) ||
        !ReadString(in, record.result.text, 64ULL * 1024ULL * 1024ULL) ||
        !ReadValue(in, record.result.image_encode_ms) ||
        !ReadValue(in, record.result.prompt_eval_ms) ||
        !ReadValue(in, record.result.eval_ms) ||
        !ReadValue(in, record.result.prompt_tokens) ||
        !ReadValue(in, record.result.generated_tokens) ||
        !ReadValue(in, record.result.created_at_ms) ||
        !ReadValue(in, record.result.last_hit_at_ms) ||
        !ReadValue(in, record.result.hit_count)) {
        return std::nullopt;
    }

    if (!LooksLikeHash(record.key.cache_key) ||
        !LooksLikeHash(record.key.image_sha256) ||
        !LooksLikeHash(record.key.prompt_sha256)) {
        return std::nullopt;
    }

    record.result.cache_key = record.key.cache_key;
    record.result.image_sha256 = record.key.image_sha256;
    record.result.prompt_sha256 = record.key.prompt_sha256;
    if (options_.store_prompts) {
        record.prompt = ReadSmallFile(PromptPath(options_, record.key.prompt_sha256), 64ULL * 1024ULL * 1024ULL);
    }
    if (options_.store_images) {
        record.image_data = ReadByteFile(ImagePath(options_, record.key.image_sha256), options_.max_bytes);
    }

    Entry entry;
    entry.bytes = EstimateBytes(record);
    entry.record = std::move(record);
    return entry;
}

void VLMCache::RemovePersistedResult(const std::string& cache_key) {
    if (!options_.persist || !LooksLikeHash(cache_key)) {
        return;
    }
    std::error_code ec;
    std::filesystem::remove(ResultPath(options_, cache_key), ec);
}

}
