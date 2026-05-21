#pragma once
#include <string>

namespace agent::net {

/// Minimum TLS protocol version.
enum class TlsMinVersion {
    Tls12,
    Tls13,
};

/// How to verify the peer's certificate during the handshake.
enum class TlsVerifyMode {
    /// Verify against the system default trust store (recommended for production).
    Default,
    /// Skip verification entirely.  Test-only — exposes you to MITM.
    None,
};

/// Options for building a client-side TLS context.
struct TlsClientOptions {
    TlsMinVersion min_version = TlsMinVersion::Tls12;
    TlsVerifyMode verify_mode = TlsVerifyMode::Default;
    /// Optional path to a custom CA bundle (PEM).  Empty = use system defaults.
    std::string ca_bundle_path;
};

}  // namespace agent::net
