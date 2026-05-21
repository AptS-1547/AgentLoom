#include "vector_fingerprint.h"

#include <openssl/evp.h>
#include <openssl/sha.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

namespace agent::vector_storage::VectorFingerprint {

namespace {

std::string ToHex(const unsigned char* data, std::size_t len) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.resize(len * 2);
    for (std::size_t i = 0; i < len; ++i) {
        out[2 * i]     = kHex[(data[i] >> 4) & 0x0F];
        out[2 * i + 1] = kHex[data[i] & 0x0F];
    }
    return out;
}

std::string Sha256Hex(const std::string& payload) {
    std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
    unsigned int hash_len = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) return {};
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, payload.data(), payload.size()) != 1 ||
        EVP_DigestFinal_ex(ctx, hash.data(), &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        return {};
    }
    EVP_MD_CTX_free(ctx);
    return ToHex(hash.data(), hash_len);
}

std::string Sha256OfFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) return {};
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        return {};
    }

    constexpr std::size_t kBufSize = 64 * 1024;
    std::vector<char> buffer(kBufSize);
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        std::streamsize got = file.gcount();
        if (got > 0) {
            if (EVP_DigestUpdate(ctx, buffer.data(), static_cast<std::size_t>(got)) != 1) {
                EVP_MD_CTX_free(ctx);
                return {};
            }
        }
    }

    std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
    unsigned int hash_len = 0;
    if (EVP_DigestFinal_ex(ctx, hash.data(), &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        return {};
    }
    EVP_MD_CTX_free(ctx);
    return ToHex(hash.data(), hash_len);
}

}  // namespace

std::string ComputeEmbeddingFingerprint(const std::string& model_path,
                                        std::size_t dimension,
                                        const std::string& pooling_strategy,
                                        const std::string& normalization) {
    std::ostringstream oss;
    oss << "embed-v1\n"
        << model_path << '\n'
        << dimension << '\n'
        << pooling_strategy << '\n'
        << normalization;
    return Sha256Hex(oss.str());
}

std::string ComputeTokenizerFingerprint(const std::string& tokenizer_json_path) {
    return Sha256OfFile(tokenizer_json_path);
}

}  // namespace agent::vector_storage::VectorFingerprint
