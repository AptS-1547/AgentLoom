#include "gateway_auth.h"

#include <nlohmann/json.hpp>

#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/rsa.h>

#include "sqlite/sqlite_connection.h"
#include "sqlite/sqlite_statement.h"
#include "redis_connection_pool.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <vector>

namespace agent::service::gateway {
namespace {

using Json = nlohmann::json;
using storage::sqlite::SqliteConnection;
using storage::sqlite::SqliteStepResult;

constexpr int kPasswordIterations = 120000;
constexpr std::size_t kPasswordSaltBytes = 16;
constexpr std::size_t kPasswordHashBytes = 32;

struct BioDeleter {
    void operator()(BIO* bio) const noexcept {
        BIO_free(bio);
    }
};

struct PKeyDeleter {
    void operator()(EVP_PKEY* key) const noexcept {
        EVP_PKEY_free(key);
    }
};

struct DigestCtxDeleter {
    void operator()(EVP_MD_CTX* ctx) const noexcept {
        EVP_MD_CTX_free(ctx);
    }
};

std::string Trim(std::string_view value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

std::string HexEncode(std::span<const unsigned char> bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned char byte : bytes) {
        out << std::setw(2) << static_cast<int>(byte);
    }
    return out.str();
}

core::Result<std::string> HashPassword(std::string_view password,
                                       std::string_view salt,
                                       int iterations) {
    if (password.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "password is required");
    }
    if (salt.empty() || iterations <= 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "password hash parameters are invalid");
    }
    std::array<unsigned char, kPasswordHashBytes> hash{};
    const int ok = PKCS5_PBKDF2_HMAC(
        password.data(),
        static_cast<int>(password.size()),
        reinterpret_cast<const unsigned char*>(salt.data()),
        static_cast<int>(salt.size()),
        iterations,
        EVP_sha256(),
        static_cast<int>(hash.size()),
        hash.data());
    if (ok != 1) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to hash password");
    }
    return HexEncode(hash);
}

core::Result<AuthUserRecord> BuildPasswordRecord(const AuthRegistrationRequest& request) {
    AuthUserRecord record;
    record.username = Trim(request.username);
    if (record.username.empty() && request.password.empty()) {
        return record;
    }
    if (record.username.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "username is required");
    }
    if (request.password.size() < 8) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "password must contain at least 8 bytes");
    }

    std::array<unsigned char, kPasswordSaltBytes> salt{};
    if (RAND_bytes(salt.data(), static_cast<int>(salt.size())) != 1) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to generate password salt");
    }
    record.password_salt = HexEncode(salt);
    record.password_iterations = kPasswordIterations;
    auto hash = HashPassword(request.password, record.password_salt, record.password_iterations);
    if (!hash.ok()) {
        return hash.status();
    }
    record.password_hash = std::move(hash).value();
    return record;
}

core::Status VerifyPassword(std::string_view password, const AuthUserRecord& record) {
    if (record.username.empty() || record.password_hash.empty() ||
        record.password_salt.empty() || record.password_iterations <= 0) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "password login is not configured for this user");
    }
    auto hash = HashPassword(password, record.password_salt, record.password_iterations);
    if (!hash.ok()) {
        return hash.status();
    }
    return hash.value() == record.password_hash
        ? core::Status::Ok()
        : core::Status::Error(core::ErrorCode::PermissionDenied, "invalid username or password");
}

std::vector<std::string_view> SplitJwt(std::string_view token) {
    std::vector<std::string_view> parts;
    std::size_t start = 0;
    while (start <= token.size()) {
        const auto dot = token.find('.', start);
        if (dot == std::string_view::npos) {
            parts.push_back(token.substr(start));
            break;
        }
        parts.push_back(token.substr(start, dot - start));
        start = dot + 1;
    }
    return parts;
}

int Base64UrlValue(char ch) {
    if (ch >= 'A' && ch <= 'Z') {
        return ch - 'A';
    }
    if (ch >= 'a' && ch <= 'z') {
        return ch - 'a' + 26;
    }
    if (ch >= '0' && ch <= '9') {
        return ch - '0' + 52;
    }
    if (ch == '-' || ch == '+') {
        return 62;
    }
    if (ch == '_' || ch == '/') {
        return 63;
    }
    return -1;
}

core::Result<std::string> Base64UrlDecode(std::string_view input) {
    std::string out;
    out.reserve(input.size() * 3 / 4);
    int value = 0;
    int bits = -8;
    for (char ch : input) {
        if (ch == '=') {
            break;
        }
        const int decoded = Base64UrlValue(ch);
        if (decoded < 0) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "invalid base64url character");
        }
        value = (value << 6) | decoded;
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<char>((value >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

std::string Base64UrlEncode(std::string_view input) {
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    out.reserve((input.size() + 2) / 3 * 4);
    std::uint32_t value = 0;
    int bits = -6;
    for (unsigned char ch : input) {
        value = (value << 8) | ch;
        bits += 8;
        while (bits >= 0) {
            out.push_back(alphabet[(value >> bits) & 0x3F]);
            bits -= 6;
        }
    }
    if (bits > -6) {
        out.push_back(alphabet[((value << 8) >> (bits + 8)) & 0x3F]);
    }
    return out;
}

std::string HeaderValue(const ::net::BeastHttpRequest& req, ::net::http::field field) {
    auto it = req.find(field);
    if (it == req.end()) {
        return {};
    }
    return std::string(it->value());
}

std::string HeaderValue(const ::net::BeastHttpRequest& req, std::string_view field) {
    auto it = req.find(field);
    if (it == req.end()) {
        return {};
    }
    return std::string(it->value());
}

core::Status VerifyRs256(std::string_view signing_input,
                         std::string_view signature,
                         std::string_view public_key_pem) {
    if (public_key_pem.empty()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "JWT public key is not configured");
    }

    std::unique_ptr<BIO, BioDeleter> bio(BIO_new_mem_buf(public_key_pem.data(), static_cast<int>(public_key_pem.size())));
    if (!bio) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to allocate OpenSSL BIO");
    }
    std::unique_ptr<EVP_PKEY, PKeyDeleter> key(PEM_read_bio_PUBKEY(bio.get(), nullptr, nullptr, nullptr));
    if (!key) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "failed to parse JWT public key");
    }
    std::unique_ptr<EVP_MD_CTX, DigestCtxDeleter> ctx(EVP_MD_CTX_new());
    if (!ctx) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to allocate OpenSSL digest context");
    }
    if (EVP_DigestVerifyInit(ctx.get(), nullptr, EVP_sha256(), nullptr, key.get()) != 1) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to initialize JWT verifier");
    }
    if (EVP_DigestVerifyUpdate(ctx.get(), signing_input.data(), signing_input.size()) != 1) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to update JWT verifier");
    }
    const int ok = EVP_DigestVerifyFinal(
        ctx.get(),
        reinterpret_cast<const unsigned char*>(signature.data()),
        signature.size());
    if (ok != 1) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "invalid JWT signature");
    }
    return core::Status::Ok();
}

core::Result<std::string> SignRs256(std::string_view signing_input, std::string_view private_key_pem) {
    if (private_key_pem.empty()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "JWT private key is not configured");
    }

    std::unique_ptr<BIO, BioDeleter> bio(BIO_new_mem_buf(private_key_pem.data(), static_cast<int>(private_key_pem.size())));
    if (!bio) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to allocate OpenSSL BIO");
    }
    std::unique_ptr<EVP_PKEY, PKeyDeleter> key(PEM_read_bio_PrivateKey(bio.get(), nullptr, nullptr, nullptr));
    if (!key) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "failed to parse JWT private key");
    }
    std::unique_ptr<EVP_MD_CTX, DigestCtxDeleter> ctx(EVP_MD_CTX_new());
    if (!ctx) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to allocate OpenSSL digest context");
    }
    if (EVP_DigestSignInit(ctx.get(), nullptr, EVP_sha256(), nullptr, key.get()) != 1) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to initialize JWT signer");
    }
    if (EVP_DigestSignUpdate(ctx.get(), signing_input.data(), signing_input.size()) != 1) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to update JWT signer");
    }
    std::size_t signature_size = 0;
    if (EVP_DigestSignFinal(ctx.get(), nullptr, &signature_size) != 1 || signature_size == 0) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to size JWT signature");
    }
    std::string signature(signature_size, '\0');
    if (EVP_DigestSignFinal(
            ctx.get(),
            reinterpret_cast<unsigned char*>(signature.data()),
            &signature_size) != 1) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to sign JWT");
    }
    signature.resize(signature_size);
    return signature;
}

std::string GenerateUuidV4() {
    std::array<unsigned char, 16> bytes{};
    std::random_device rd;
    for (auto& byte : bytes) {
        byte = static_cast<unsigned char>(rd());
    }
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0F) | 0x40);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3F) | 0x80);

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            out << '-';
        }
        out << std::setw(2) << static_cast<int>(bytes[i]);
    }
    return out.str();
}

std::string JsonStringClaim(const Json& payload, std::string_view name) {
    auto it = payload.find(std::string(name));
    if (it == payload.end()) {
        return {};
    }
    if (it->is_string()) {
        return it->get<std::string>();
    }
    return {};
}

std::int64_t ToUnixSeconds(std::chrono::system_clock::time_point time) {
    return std::chrono::duration_cast<std::chrono::seconds>(time.time_since_epoch()).count();
}

std::chrono::system_clock::time_point FromUnixSeconds(std::int64_t seconds) {
    return std::chrono::system_clock::time_point(std::chrono::seconds(seconds));
}

Json AuthSessionRecordToJson(const AuthSessionRecord& record) {
    return Json{
        {"token_id", record.token_id},
        {"user_uuid", record.user_uuid},
        {"tenant_id", record.tenant_id.empty() ? "default" : record.tenant_id},
        {"subject", record.subject},
        {"issued_at", ToUnixSeconds(record.issued_at)},
        {"expires_at", ToUnixSeconds(record.expires_at)},
        {"revoked", record.revoked},
    };
}

core::Result<AuthSessionRecord> AuthSessionRecordFromJson(std::string_view payload) {
    Json json;
    try {
        json = Json::parse(payload);
    } catch (const Json::exception& e) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, e.what());
    }

    AuthSessionRecord record;
    record.token_id = json.value("token_id", std::string{});
    record.user_uuid = json.value("user_uuid", std::string{});
    record.tenant_id = json.value("tenant_id", std::string{"default"});
    record.subject = json.value("subject", std::string{});
    record.issued_at = FromUnixSeconds(json.value("issued_at", std::int64_t{0}));
    record.expires_at = FromUnixSeconds(json.value("expires_at", std::int64_t{0}));
    record.revoked = json.value("revoked", false);
    if (record.token_id.empty() || record.user_uuid.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "invalid auth session record");
    }
    return record;
}

Json AuthUserRecordToJson(const AuthUserRecord& record) {
    return Json{
        {"user_uuid", record.user_uuid},
        {"tenant_id", record.tenant_id.empty() ? "default" : record.tenant_id},
        {"username", record.username},
        {"password_hash", record.password_hash},
        {"password_salt", record.password_salt},
        {"password_iterations", record.password_iterations},
        {"subject", record.subject},
        {"created_at", ToUnixSeconds(record.created_at)},
        {"updated_at", ToUnixSeconds(record.updated_at)},
        {"disabled", record.disabled},
    };
}

core::Result<AuthUserRecord> AuthUserRecordFromJson(std::string_view payload) {
    Json json;
    try {
        json = Json::parse(payload);
    } catch (const Json::exception& e) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, e.what());
    }

    AuthUserRecord record;
    record.user_uuid = json.value("user_uuid", std::string{});
    record.tenant_id = json.value("tenant_id", std::string{"default"});
    record.username = json.value("username", std::string{});
    record.password_hash = json.value("password_hash", std::string{});
    record.password_salt = json.value("password_salt", std::string{});
    record.password_iterations = json.value("password_iterations", 0);
    record.subject = json.value("subject", std::string{});
    record.created_at = FromUnixSeconds(json.value("created_at", std::int64_t{0}));
    record.updated_at = FromUnixSeconds(json.value("updated_at", std::int64_t{0}));
    record.disabled = json.value("disabled", false);
    if (record.user_uuid.empty() || record.username.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "invalid auth user record");
    }
    return record;
}

std::chrono::seconds RemainingTtl(std::chrono::system_clock::time_point expires_at) {
    const auto now = std::chrono::system_clock::now();
    if (expires_at <= now) {
        return std::chrono::seconds(1);
    }
    return std::max(std::chrono::seconds(1), std::chrono::duration_cast<std::chrono::seconds>(expires_at - now));
}

bool AudienceMatches(const Json& payload, std::string_view expected) {
    if (expected.empty()) {
        return true;
    }
    auto it = payload.find("aud");
    if (it == payload.end()) {
        return false;
    }
    if (it->is_string()) {
        return it->get<std::string>() == expected;
    }
    if (it->is_array()) {
        return std::any_of(it->begin(), it->end(), [&](const Json& value) {
            return value.is_string() && value.get<std::string>() == expected;
        });
    }
    return false;
}

core::Status EnsureColumn(SqliteConnection& connection,
                          std::string_view table,
                          std::string_view column,
                          std::string_view definition) {
    auto statement_result = connection.Prepare("PRAGMA table_info(" + std::string(table) + ")");
    if (!statement_result.ok()) {
        return statement_result.status();
    }
    auto statement = std::move(statement_result).value();
    while (true) {
        auto step = statement.Step();
        if (!step.ok()) {
            return step.status();
        }
        if (step.value() == SqliteStepResult::Done) {
            break;
        }
        if (statement.ColumnText(1) == column) {
            return core::Status::Ok();
        }
    }
    return connection.Execute(
        "ALTER TABLE " + std::string(table) + " ADD COLUMN " + std::string(definition));
}

} // namespace

JwtCookieAuthenticator::JwtCookieAuthenticator(GatewayAuthOptions options,
                                               std::shared_ptr<IAuthSessionStore> session_store,
                                               core::LoggerAdapter logger)
    : options_(std::move(options)),
      session_store_(std::move(session_store)),
      logger_(std::move(logger)) {}

core::Result<AuthIdentity> JwtCookieAuthenticator::Authenticate(const ::net::BeastHttpRequest& request) const {
    if (!options_.enabled) {
        AuthIdentity identity;
        identity.authenticated = false;
        return identity;
    }

    auto token = ExtractToken(request);
    if (!token.ok()) {
        if (options_.allow_dev_identity && !options_.require_auth_for_api) {
            AuthIdentity identity;
            identity.authenticated = false;
            return identity;
        }
        return token.status();
    }
    auto verified = VerifyJwt(token.value());
    if (!verified.ok()) {
        return verified.status();
    }
    return ResolveLocalSession(verified.value());
}

core::Result<std::string> JwtCookieAuthenticator::ExtractToken(const ::net::BeastHttpRequest& request) const {
    const auto auth = HeaderValue(request, ::net::http::field::authorization);
    constexpr std::string_view bearer_prefix = "Bearer ";
    if (auth.size() > bearer_prefix.size() &&
        std::equal(bearer_prefix.begin(), bearer_prefix.end(), auth.begin(), [](char lhs, char rhs) {
            return std::tolower(static_cast<unsigned char>(lhs)) == std::tolower(static_cast<unsigned char>(rhs));
        })) {
        auto token = Trim(std::string_view(auth).substr(bearer_prefix.size()));
        if (!token.empty()) {
            return token;
        }
    }

    const auto cookie = HeaderValue(request, ::net::http::field::cookie);
    if (auto value = ExtractCookieValue(cookie, options_.cookie_name)) {
        return *value;
    }
    return core::Status::Error(core::ErrorCode::PermissionDenied, "auth token is missing");
}

core::Result<AuthIdentity> JwtCookieAuthenticator::VerifyJwt(std::string_view token) const {
    const auto parts = SplitJwt(token);
    if (parts.size() != 3 || parts[0].empty() || parts[1].empty() || parts[2].empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "malformed JWT");
    }

    auto header_text = Base64UrlDecode(parts[0]);
    auto payload_text = Base64UrlDecode(parts[1]);
    auto signature = Base64UrlDecode(parts[2]);
    if (!header_text.ok()) {
        return header_text.status();
    }
    if (!payload_text.ok()) {
        return payload_text.status();
    }
    if (!signature.ok()) {
        return signature.status();
    }

    Json header;
    Json payload;
    try {
        header = Json::parse(header_text.value());
        payload = Json::parse(payload_text.value());
    } catch (const Json::exception& e) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, e.what());
    }

    if (header.value("alg", std::string{}) != "RS256") {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "unsupported JWT algorithm");
    }
    const std::string signing_input = std::string(parts[0]) + "." + std::string(parts[1]);
    auto verify = VerifyRs256(signing_input, signature.value(), options_.public_key_pem);
    if (!verify.ok()) {
        return verify;
    }

    const auto now = std::chrono::system_clock::now();
    const auto skew = options_.clock_skew;
    if (payload.contains("exp") && payload["exp"].is_number_integer()) {
        const auto exp = std::chrono::system_clock::time_point(std::chrono::seconds(payload["exp"].get<std::int64_t>()));
        if (now - skew > exp) {
            return core::Status::Error(core::ErrorCode::PermissionDenied, "JWT is expired");
        }
    } else {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "JWT exp claim is required");
    }
    if (payload.contains("nbf") && payload["nbf"].is_number_integer()) {
        const auto nbf = std::chrono::system_clock::time_point(std::chrono::seconds(payload["nbf"].get<std::int64_t>()));
        if (now + skew < nbf) {
            return core::Status::Error(core::ErrorCode::PermissionDenied, "JWT is not valid yet");
        }
    }
    if (!options_.issuer.empty() && JsonStringClaim(payload, "iss") != options_.issuer) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "JWT issuer mismatch");
    }
    if (!AudienceMatches(payload, options_.audience)) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "JWT audience mismatch");
    }

    AuthIdentity identity;
    identity.user_uuid = JsonStringClaim(payload, "uuid");
    if (identity.user_uuid.empty()) {
        identity.user_uuid = JsonStringClaim(payload, "user_uuid");
    }
    if (identity.user_uuid.empty()) {
        identity.user_uuid = JsonStringClaim(payload, "sub");
    }
    identity.tenant_id = JsonStringClaim(payload, "tenant");
    if (identity.tenant_id.empty()) {
        identity.tenant_id = JsonStringClaim(payload, "tenant_id");
    }
    if (identity.tenant_id.empty()) {
        identity.tenant_id = "default";
    }
    identity.subject = JsonStringClaim(payload, "sub");
    identity.issuer = JsonStringClaim(payload, "iss");
    identity.audience = options_.audience;
    identity.token_id = JsonStringClaim(payload, "jti");
    if (identity.token_id.empty()) {
        identity.token_id = JsonStringClaim(payload, "sid");
    }
    identity.expires_at = std::chrono::system_clock::time_point(std::chrono::seconds(payload["exp"].get<std::int64_t>()));
    identity.authenticated = true;

    if (identity.user_uuid.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "JWT uuid/sub claim is required");
    }
    return identity;
}

core::Result<AuthIdentity> JwtCookieAuthenticator::ResolveLocalSession(const AuthIdentity& jwt_identity) const {
    if (!session_store_) {
        return jwt_identity;
    }
    if (jwt_identity.token_id.empty()) {
        if (options_.require_session_record) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "JWT jti/sid claim is required");
        }
        return jwt_identity;
    }

    auto resolved = session_store_->ResolveSession(jwt_identity.token_id);
    if (!resolved.ok()) {
        if (resolved.status().code() != core::ErrorCode::NotFound ||
            options_.require_session_record ||
            !options_.auto_provision_session) {
            return resolved.status();
        }
        AuthSessionRecord record;
        record.token_id = jwt_identity.token_id;
        record.user_uuid = jwt_identity.user_uuid;
        record.tenant_id = jwt_identity.tenant_id;
        record.subject = jwt_identity.subject;
        record.issued_at = std::chrono::system_clock::now();
        record.expires_at = jwt_identity.expires_at;
        auto upsert = session_store_->UpsertSession(record);
        if (!upsert.ok()) {
            return upsert;
        }
        return jwt_identity;
    }

    const auto& record = resolved.value();
    if (record.revoked) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "auth session is revoked");
    }
    if (std::chrono::system_clock::now() - options_.clock_skew > record.expires_at) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "auth session is expired");
    }

    AuthIdentity identity = jwt_identity;
    identity.user_uuid = record.user_uuid;
    identity.tenant_id = record.tenant_id;
    identity.subject = record.subject.empty() ? jwt_identity.subject : record.subject;
    identity.expires_at = std::min(jwt_identity.expires_at, record.expires_at);
    return identity;
}

JwtAuthRegistrationService::JwtAuthRegistrationService(GatewayAuthOptions options,
                                                       std::shared_ptr<IAuthSessionStore> session_store,
                                                       core::LoggerAdapter logger)
    : options_(std::move(options)),
      session_store_(std::move(session_store)),
      logger_(std::move(logger)) {}

core::Result<AuthRegistrationResult> JwtAuthRegistrationService::Register(const AuthRegistrationRequest& request) {
    if (!options_.enabled) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "gateway auth is not enabled");
    }
    if (!session_store_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "auth session store is not configured");
    }

    const auto now = std::chrono::system_clock::now();
    const auto ttl = request.ttl.count() > 0 ? request.ttl : options_.token_ttl;
    if (ttl.count() <= 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "auth token ttl must be positive");
    }

    AuthIdentity identity;
    identity.user_uuid = request.user_uuid.empty() ? GenerateUuidV4() : request.user_uuid;
    identity.tenant_id = request.tenant_id.empty() ? std::string("default") : request.tenant_id;
    identity.subject = request.subject.empty() ? identity.user_uuid : request.subject;
    identity.issuer = options_.issuer;
    identity.audience = options_.audience;
    identity.token_id = GenerateUuidV4();
    identity.expires_at = now + ttl;
    identity.authenticated = true;

    auto password_record = BuildPasswordRecord(request);
    if (!password_record.ok()) {
        return password_record.status();
    }
    if (!password_record.value().username.empty()) {
        AuthUserRecord user = std::move(password_record).value();
        user.user_uuid = identity.user_uuid;
        user.tenant_id = identity.tenant_id;
        user.subject = identity.subject;
        user.created_at = now;
        user.updated_at = now;
        auto upsert_user = session_store_->UpsertUser(user);
        if (!upsert_user.ok()) {
            return upsert_user;
        }
    }

    AuthSessionRecord session;
    session.token_id = identity.token_id;
    session.user_uuid = identity.user_uuid;
    session.tenant_id = identity.tenant_id;
    session.subject = identity.subject;
    session.issued_at = now;
    session.expires_at = identity.expires_at;
    auto upsert = session_store_->UpsertSession(session);
    if (!upsert.ok()) {
        return upsert;
    }

    return IssueForIdentity(identity, now);
}

core::Result<AuthRegistrationResult> JwtAuthRegistrationService::Login(const AuthLoginRequest& request) {
    if (!options_.enabled) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "gateway auth is not enabled");
    }
    if (!session_store_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "auth session store is not configured");
    }

    const auto username = Trim(request.username);
    if (username.empty() || request.password.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "username and password are required");
    }

    auto user = session_store_->ResolveUserByUsername(username);
    if (!user.ok()) {
        return user.status();
    }
    if (user.value().disabled) {
        return core::Status::Error(core::ErrorCode::PermissionDenied, "auth user is disabled");
    }
    if (auto verified = VerifyPassword(request.password, user.value()); !verified.ok()) {
        return verified;
    }

    const auto now = std::chrono::system_clock::now();
    const auto ttl = request.ttl.count() > 0 ? request.ttl : options_.token_ttl;
    if (ttl.count() <= 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "auth token ttl must be positive");
    }

    AuthIdentity identity;
    identity.user_uuid = user.value().user_uuid;
    identity.tenant_id = user.value().tenant_id.empty() ? std::string("default") : user.value().tenant_id;
    identity.subject = user.value().subject.empty() ? user.value().user_uuid : user.value().subject;
    identity.issuer = options_.issuer;
    identity.audience = options_.audience;
    identity.token_id = GenerateUuidV4();
    identity.expires_at = now + ttl;
    identity.authenticated = true;

    AuthSessionRecord session;
    session.token_id = identity.token_id;
    session.user_uuid = identity.user_uuid;
    session.tenant_id = identity.tenant_id;
    session.subject = identity.subject;
    session.issued_at = now;
    session.expires_at = identity.expires_at;
    session.revoked = false;
    auto upsert = session_store_->UpsertSession(session);
    if (!upsert.ok()) {
        return upsert;
    }

    return IssueForIdentity(identity, now);
}

core::Result<AuthRegistrationResult> JwtAuthRegistrationService::IssueForIdentity(
    const AuthIdentity& identity,
    std::chrono::system_clock::time_point issued_at) const {
    auto token = IssueJwt(identity, issued_at);
    if (!token.ok()) {
        return token.status();
    }

    AuthRegistrationResult result;
    result.identity = identity;
    result.token = std::move(token).value();
    result.cookie_header = BuildCookieHeader(result.token, result.identity.expires_at);
    result.issued_at = issued_at;
    return result;
}

core::Result<std::string> JwtAuthRegistrationService::IssueJwt(
    const AuthIdentity& identity,
    std::chrono::system_clock::time_point issued_at) const {
    Json header{{"alg", "RS256"}, {"typ", "JWT"}};
    Json payload{
        {"sub", identity.subject.empty() ? identity.user_uuid : identity.subject},
        {"uuid", identity.user_uuid},
        {"user_uuid", identity.user_uuid},
        {"tenant", identity.tenant_id},
        {"tenant_id", identity.tenant_id},
        {"jti", identity.token_id},
        {"sid", identity.token_id},
        {"iat", ToUnixSeconds(issued_at)},
        {"exp", ToUnixSeconds(identity.expires_at)},
    };
    if (!options_.issuer.empty()) {
        payload["iss"] = options_.issuer;
    }
    if (!options_.audience.empty()) {
        payload["aud"] = options_.audience;
    }

    const auto header_part = Base64UrlEncode(header.dump());
    const auto payload_part = Base64UrlEncode(payload.dump());
    const auto signing_input = header_part + "." + payload_part;
    auto signature = SignRs256(signing_input, options_.private_key_pem);
    if (!signature.ok()) {
        return signature.status();
    }
    return signing_input + "." + Base64UrlEncode(signature.value());
}

std::string JwtAuthRegistrationService::BuildCookieHeader(
    std::string_view token,
    std::chrono::system_clock::time_point expires_at) const {
    std::ostringstream out;
    out << options_.cookie_name << '=' << token
        << "; Path=/"
        << "; Max-Age=" << std::max<std::int64_t>(0, ToUnixSeconds(expires_at) - ToUnixSeconds(std::chrono::system_clock::now()));
    if (options_.cookie_http_only) {
        out << "; HttpOnly";
    }
    if (options_.cookie_secure) {
        out << "; Secure";
    }
    if (!options_.cookie_same_site.empty()) {
        out << "; SameSite=" << options_.cookie_same_site;
    }
    return out.str();
}

SqliteAuthSessionStore::SqliteAuthSessionStore(std::string database_path)
    : database_path_(std::move(database_path)) {}

core::Status SqliteAuthSessionStore::EnsureSchema() {
    auto connection_result = SqliteConnection::Open(database_path_);
    if (!connection_result.ok()) {
        return connection_result.status();
    }
    auto connection = std::move(connection_result).value();
    auto wal = connection.EnableWal();
    if (!wal.ok()) {
        return wal;
    }
    auto create = connection.Execute(
        "CREATE TABLE IF NOT EXISTS gateway_auth_sessions ("
        "token_id TEXT PRIMARY KEY,"
        "user_uuid TEXT NOT NULL,"
        "tenant_id TEXT NOT NULL,"
        "subject TEXT,"
        "issued_at INTEGER NOT NULL,"
        "expires_at INTEGER NOT NULL,"
        "revoked INTEGER NOT NULL DEFAULT 0,"
        "revoked_reason TEXT,"
        "updated_at INTEGER NOT NULL"
        ")");
    if (!create.ok()) return create;
    auto create_users = connection.Execute(
        "CREATE TABLE IF NOT EXISTS gateway_auth_users ("
        "user_uuid TEXT PRIMARY KEY,"
        "tenant_id TEXT NOT NULL,"
        "username TEXT NOT NULL UNIQUE,"
        "password_hash TEXT NOT NULL,"
        "password_salt TEXT NOT NULL,"
        "password_iterations INTEGER NOT NULL,"
        "subject TEXT,"
        "created_at INTEGER NOT NULL,"
        "updated_at INTEGER NOT NULL,"
        "disabled INTEGER NOT NULL DEFAULT 0"
        ")");
    if (!create_users.ok()) return create_users;
    if (auto status = EnsureColumn(connection, "gateway_auth_users", "disabled", "disabled INTEGER NOT NULL DEFAULT 0"); !status.ok()) return status;
    return core::Status::Ok();
}

core::Result<AuthSessionRecord> SqliteAuthSessionStore::ResolveSession(std::string_view token_id) {
    auto connection_result = SqliteConnection::Open(database_path_);
    if (!connection_result.ok()) {
        return connection_result.status();
    }
    auto connection = std::move(connection_result).value();
    auto statement_result = connection.Prepare(
        "SELECT token_id,user_uuid,tenant_id,subject,issued_at,expires_at,revoked "
        "FROM gateway_auth_sessions WHERE token_id=?1");
    if (!statement_result.ok()) {
        return statement_result.status();
    }
    auto statement = std::move(statement_result).value();
    auto bind = statement.BindText(1, std::string(token_id));
    if (!bind.ok()) {
        return bind;
    }
    auto step = statement.Step();
    if (!step.ok()) {
        return step.status();
    }
    if (step.value() != SqliteStepResult::Row) {
        return core::Status::Error(core::ErrorCode::NotFound, "auth session not found");
    }
    AuthSessionRecord record;
    record.token_id = statement.ColumnText(0);
    record.user_uuid = statement.ColumnText(1);
    record.tenant_id = statement.ColumnText(2);
    record.subject = statement.ColumnText(3);
    record.issued_at = FromUnixSeconds(statement.ColumnInt64(4));
    record.expires_at = FromUnixSeconds(statement.ColumnInt64(5));
    record.revoked = statement.ColumnInt(6) != 0;
    return record;
}

core::Result<AuthUserRecord> SqliteAuthSessionStore::ResolveUserByUsername(std::string_view username) {
    auto connection_result = SqliteConnection::Open(database_path_);
    if (!connection_result.ok()) {
        return connection_result.status();
    }
    auto connection = std::move(connection_result).value();
    auto statement_result = connection.Prepare(
        "SELECT user_uuid,tenant_id,username,password_hash,password_salt,password_iterations,subject,created_at,updated_at,disabled "
        "FROM gateway_auth_users WHERE username=?1");
    if (!statement_result.ok()) {
        return statement_result.status();
    }
    auto statement = std::move(statement_result).value();
    if (auto status = statement.BindText(1, std::string(username)); !status.ok()) {
        return status;
    }
    auto step = statement.Step();
    if (!step.ok()) {
        return step.status();
    }
    if (step.value() != SqliteStepResult::Row) {
        return core::Status::Error(core::ErrorCode::NotFound, "auth user not found");
    }
    AuthUserRecord record;
    record.user_uuid = statement.ColumnText(0);
    record.tenant_id = statement.ColumnText(1);
    record.username = statement.ColumnText(2);
    record.password_hash = statement.ColumnText(3);
    record.password_salt = statement.ColumnText(4);
    record.password_iterations = statement.ColumnInt(5);
    record.subject = statement.ColumnText(6);
    record.created_at = FromUnixSeconds(statement.ColumnInt64(7));
    record.updated_at = FromUnixSeconds(statement.ColumnInt64(8));
    record.disabled = statement.ColumnInt(9) != 0;
    return record;
}

core::Status SqliteAuthSessionStore::UpsertUser(const AuthUserRecord& record) {
    if (record.user_uuid.empty() || record.username.empty() ||
        record.password_hash.empty() || record.password_salt.empty() || record.password_iterations <= 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "auth user record is incomplete");
    }
    auto connection_result = SqliteConnection::Open(database_path_);
    if (!connection_result.ok()) {
        return connection_result.status();
    }
    auto connection = std::move(connection_result).value();
    auto existing_result = connection.Prepare("SELECT user_uuid FROM gateway_auth_users WHERE username=?1");
    if (!existing_result.ok()) {
        return existing_result.status();
    }
    auto existing = std::move(existing_result).value();
    if (auto status = existing.BindText(1, record.username); !status.ok()) {
        return status;
    }
    auto existing_step = existing.Step();
    if (!existing_step.ok()) {
        return existing_step.status();
    }
    if (existing_step.value() == SqliteStepResult::Row) {
        return core::Status::Error(core::ErrorCode::AlreadyExists, "auth user already exists");
    }

    auto statement_result = connection.Prepare(
        "INSERT INTO gateway_auth_users(user_uuid,tenant_id,username,password_hash,password_salt,password_iterations,subject,created_at,updated_at,disabled) "
        "VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)");
    if (!statement_result.ok()) {
        return statement_result.status();
    }
    auto statement = std::move(statement_result).value();
    const auto now = ToUnixSeconds(std::chrono::system_clock::now());
    if (auto status = statement.BindText(1, record.user_uuid); !status.ok()) return status;
    if (auto status = statement.BindText(2, record.tenant_id.empty() ? std::string("default") : record.tenant_id); !status.ok()) return status;
    if (auto status = statement.BindText(3, record.username); !status.ok()) return status;
    if (auto status = statement.BindText(4, record.password_hash); !status.ok()) return status;
    if (auto status = statement.BindText(5, record.password_salt); !status.ok()) return status;
    if (auto status = statement.BindInt(6, record.password_iterations); !status.ok()) return status;
    if (auto status = statement.BindText(7, record.subject); !status.ok()) return status;
    if (auto status = statement.BindInt64(8, ToUnixSeconds(record.created_at) == 0 ? now : ToUnixSeconds(record.created_at)); !status.ok()) return status;
    if (auto status = statement.BindInt64(9, ToUnixSeconds(record.updated_at) == 0 ? now : ToUnixSeconds(record.updated_at)); !status.ok()) return status;
    if (auto status = statement.BindInt(10, record.disabled ? 1 : 0); !status.ok()) return status;
    auto step = statement.Step();
    if (!step.ok()) {
        return step.status();
    }
    return core::Status::Ok();
}

core::Status SqliteAuthSessionStore::UpsertSession(const AuthSessionRecord& record) {
    if (record.token_id.empty() || record.user_uuid.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "token_id and user_uuid are required");
    }
    auto connection_result = SqliteConnection::Open(database_path_);
    if (!connection_result.ok()) {
        return connection_result.status();
    }
    auto connection = std::move(connection_result).value();
    auto statement_result = connection.Prepare(
        "INSERT INTO gateway_auth_sessions(token_id,user_uuid,tenant_id,subject,issued_at,expires_at,revoked,updated_at) "
        "VALUES(?1,?2,?3,?4,?5,?6,0,?7) "
        "ON CONFLICT(token_id) DO UPDATE SET "
        "user_uuid=excluded.user_uuid,tenant_id=excluded.tenant_id,subject=excluded.subject,"
        "expires_at=excluded.expires_at,revoked=0,revoked_reason=NULL,updated_at=excluded.updated_at");
    if (!statement_result.ok()) {
        return statement_result.status();
    }
    auto statement = std::move(statement_result).value();
    const auto now = ToUnixSeconds(std::chrono::system_clock::now());
    if (auto status = statement.BindText(1, record.token_id); !status.ok()) return status;
    if (auto status = statement.BindText(2, record.user_uuid); !status.ok()) return status;
    if (auto status = statement.BindText(3, record.tenant_id.empty() ? std::string("default") : record.tenant_id); !status.ok()) return status;
    if (auto status = statement.BindText(4, record.subject); !status.ok()) return status;
    if (auto status = statement.BindInt64(5, ToUnixSeconds(record.issued_at)); !status.ok()) return status;
    if (auto status = statement.BindInt64(6, ToUnixSeconds(record.expires_at)); !status.ok()) return status;
    if (auto status = statement.BindInt64(7, now); !status.ok()) return status;
    auto step = statement.Step();
    if (!step.ok()) {
        return step.status();
    }
    return core::Status::Ok();
}

core::Status SqliteAuthSessionStore::RevokeSession(std::string_view token_id, std::string_view reason) {
    auto connection_result = SqliteConnection::Open(database_path_);
    if (!connection_result.ok()) {
        return connection_result.status();
    }
    auto connection = std::move(connection_result).value();
    auto statement_result = connection.Prepare(
        "UPDATE gateway_auth_sessions SET revoked=1, revoked_reason=?2, updated_at=?3 WHERE token_id=?1");
    if (!statement_result.ok()) {
        return statement_result.status();
    }
    auto statement = std::move(statement_result).value();
    if (auto status = statement.BindText(1, std::string(token_id)); !status.ok()) return status;
    if (auto status = statement.BindText(2, std::string(reason)); !status.ok()) return status;
    if (auto status = statement.BindInt64(3, ToUnixSeconds(std::chrono::system_clock::now())); !status.ok()) return status;
    auto step = statement.Step();
    if (!step.ok()) {
        return step.status();
    }
    return connection.Changes() == 0
        ? core::Status::Error(core::ErrorCode::NotFound, "auth session not found")
        : core::Status::Ok();
}

RedisAuthSessionStore::RedisAuthSessionStore(std::shared_ptr<semantic_cache::RedisConnectionPool> redis,
                                             std::string key_prefix)
    : redis_(std::move(redis)),
      key_prefix_(std::move(key_prefix)) {
    if (key_prefix_.empty()) {
        key_prefix_ = "agent:gateway:auth";
    }
}

core::Status RedisAuthSessionStore::EnsureSchema() {
    if (!redis_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "redis auth session store is not configured");
    }
    return redis_->running()
        ? core::Status::Ok()
        : core::Status::Error(core::ErrorCode::FailedPrecondition, "redis auth session store is not running");
}

core::Result<AuthSessionRecord> RedisAuthSessionStore::ResolveSession(std::string_view token_id) {
    if (token_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "token_id is required");
    }
    auto payload = redis_->Get(SessionKey(token_id));
    if (!payload.ok()) {
        return payload.status();
    }
    return AuthSessionRecordFromJson(payload.value());
}

core::Result<AuthUserRecord> RedisAuthSessionStore::ResolveUserByUsername(std::string_view username) {
    if (username.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "username is required");
    }
    auto user_uuid = redis_->Get(UsernameKey(username));
    if (!user_uuid.ok()) {
        return user_uuid.status();
    }
    auto payload = redis_->Get(UserKey(user_uuid.value()));
    if (!payload.ok()) {
        return payload.status();
    }
    return AuthUserRecordFromJson(payload.value());
}

core::Status RedisAuthSessionStore::UpsertUser(const AuthUserRecord& record) {
    if (record.user_uuid.empty() || record.username.empty() ||
        record.password_hash.empty() || record.password_salt.empty() || record.password_iterations <= 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "auth user record is incomplete");
    }
    const auto username_key = UsernameKey(record.username);
    auto reserve_username = redis_->SetIfAbsent(username_key, record.user_uuid);
    if (!reserve_username.ok()) {
        return reserve_username.status();
    }
    if (!reserve_username.value()) {
        return core::Status::Error(core::ErrorCode::AlreadyExists, "auth user already exists");
    }
    auto set_user = redis_->Set(UserKey(record.user_uuid), AuthUserRecordToJson(record).dump());
    if (!set_user.ok()) {
        (void)redis_->Del({username_key});
        return set_user;
    }
    return core::Status::Ok();
}

core::Status RedisAuthSessionStore::UpsertSession(const AuthSessionRecord& record) {
    if (record.token_id.empty() || record.user_uuid.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "token_id and user_uuid are required");
    }
    auto ttl = RemainingTtl(record.expires_at);
    auto set_session = redis_->Set(SessionKey(record.token_id),
                                  AuthSessionRecordToJson(record).dump(),
                                  ttl);
    if (!set_session.ok()) {
        return set_session;
    }
    return core::Status::Ok();
}

core::Status RedisAuthSessionStore::RevokeSession(std::string_view token_id, std::string_view) {
    auto resolved = ResolveSession(token_id);
    if (!resolved.ok()) {
        return resolved.status();
    }
    auto record = resolved.value();
    record.revoked = true;
    return redis_->Set(SessionKey(token_id),
                       AuthSessionRecordToJson(record).dump(),
                       RemainingTtl(record.expires_at));
}

std::string RedisAuthSessionStore::SessionKey(std::string_view token_id) const {
    return key_prefix_ + ":session:" + std::string(token_id);
}

std::string RedisAuthSessionStore::UserKey(std::string_view user_uuid) const {
    return key_prefix_ + ":user:" + std::string(user_uuid);
}

std::string RedisAuthSessionStore::UsernameKey(std::string_view username) const {
    return key_prefix_ + ":username:" + std::string(username);
}

std::optional<std::string> ExtractCookieValue(std::string_view cookie_header, std::string_view name) {
    std::size_t start = 0;
    while (start < cookie_header.size()) {
        auto end = cookie_header.find(';', start);
        if (end == std::string_view::npos) {
            end = cookie_header.size();
        }
        auto item = Trim(cookie_header.substr(start, end - start));
        const auto eq = item.find('=');
        if (eq != std::string::npos) {
            auto key = Trim(std::string_view(item).substr(0, eq));
            auto value = Trim(std::string_view(item).substr(eq + 1));
            if (key == name) {
                return value;
            }
        }
        start = end + 1;
    }
    return std::nullopt;
}

core::Result<GatewayDevelopmentKeyPair> GenerateDevelopmentRsaKeyPair() {
    std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)> ctx(
        EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr),
        EVP_PKEY_CTX_free);
    if (!ctx ||
        EVP_PKEY_keygen_init(ctx.get()) != 1 ||
        EVP_PKEY_CTX_set_rsa_keygen_bits(ctx.get(), 2048) != 1) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to initialize RSA key generator");
    }

    EVP_PKEY* raw_key = nullptr;
    if (EVP_PKEY_keygen(ctx.get(), &raw_key) != 1 || raw_key == nullptr) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to generate RSA key pair");
    }
    std::unique_ptr<EVP_PKEY, PKeyDeleter> key(raw_key);

    std::unique_ptr<BIO, BioDeleter> private_bio(BIO_new(BIO_s_mem()));
    std::unique_ptr<BIO, BioDeleter> public_bio(BIO_new(BIO_s_mem()));
    if (!private_bio || !public_bio) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to allocate OpenSSL BIO");
    }
    if (PEM_write_bio_PrivateKey(private_bio.get(), key.get(), nullptr, nullptr, 0, nullptr, nullptr) != 1) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to write private key PEM");
    }
    if (PEM_write_bio_PUBKEY(public_bio.get(), key.get()) != 1) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to write public key PEM");
    }

    BUF_MEM* private_mem = nullptr;
    BUF_MEM* public_mem = nullptr;
    BIO_get_mem_ptr(private_bio.get(), &private_mem);
    BIO_get_mem_ptr(public_bio.get(), &public_mem);
    if (!private_mem || !public_mem) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to read generated key PEM");
    }
    return GatewayDevelopmentKeyPair{
        std::string(private_mem->data, private_mem->length),
        std::string(public_mem->data, public_mem->length),
    };
}

} // namespace agent::service::gateway
