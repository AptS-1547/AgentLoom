# Redis Compatibility Notes

> Critical findings about Redis port compatibility and Boost.redis library behavior  
> Date: 2026-05-26 to 2026-05-27  
> Context: L3 compression E2E testing and semantic cache development

## Executive Summary

**Windows Redis port has binary payload handling issues that cause `LRANGE` to block 
indefinitely with ~1.6KB serialized data.** WSL Redis (same version) works correctly 
with identical code and payloads. Production deployments should use Linux Redis.

## Issue Details

### Symptom

E2E test for L3 long-term memory compression hangs indefinitely on `LRANGE` command 
when fetching serialized `CacheRecord` structures (~1.6KB each, 384-dim float vector 
+ metadata).

```cpp
// This blocks forever on Windows Redis (port 6379)
boost::redis::request lrange_req;
lrange_req.push("LRANGE", key, "0", "-1");
boost::redis::response<std::vector<std::string>> lrange_resp;
auto status = redis_pool->Exec(lrange_req, lrange_resp);  // Hangs here
```

### Environment

- **Windows Redis**: Native Windows port, registered as service, auto-starts on port 6379
- **WSL Redis**: Redis running in WSL2, manually started on port 5000
- **Boost.redis**: Version 1.85 (from Boost 1.85.0)
- **Payload size**: ~1.6KB per record (384 float32 values + metadata)
- **Test scenario**: 3 records written via `RPUSH`, then read via `LRANGE`

### Test Results

| Test | Windows Redis (6379) | WSL Redis (5000) |
|------|---------------------|------------------|
| Bare connection test | ✅ Pass | ✅ Pass |
| Small string RPUSH/LRANGE | ✅ Pass | ✅ Pass |
| Binary payload E2E | ❌ Hangs | ✅ Pass |

**Bare connection test** (`redis_bare_connection_test.cpp`):
- Simple PING/PONG exchange
- Small string SET/GET
- Passes on both Windows and WSL Redis

**E2E test** (`l3_compression_e2e_test.cpp`):
- Serialize 3 `CacheRecord` structures (each ~1.6KB)
- `RPUSH` to Redis list
- `LRANGE` to retrieve all records
- **Windows Redis**: hangs indefinitely on `LRANGE`, no timeout, no error
- **WSL Redis**: completes successfully in ~1.4s (including LLM call)

### Root Cause Hypothesis

Windows Redis port likely has issues with:
1. **Binary data handling** — may not correctly handle non-UTF8 binary blobs
2. **Buffer management** — possible buffer overflow or incorrect size calculation 
   for large payloads
3. **Protocol parsing** — may misinterpret binary data as protocol markers

The issue is **not** in Boost.redis client code, as the same code works correctly 
with WSL Redis.

## Boost.redis Library Issues

### 1. KEYS Command Aborts Connection

**Issue**: `KEYS` pattern matching causes connection abort in Boost.redis 1.85.

```cpp
// This aborts the connection
boost::redis::request req;
req.push("KEYS", "cache:batch:*");
boost::redis::response<std::vector<std::string>> resp;
auto status = redis_pool->Exec(req, resp);  // Connection aborted
```

**Workaround**: Use `SCAN` with `generic_response` instead:

```cpp
std::string cursor = "0";
do {
    boost::redis::request req;
    req.push("SCAN", cursor, "MATCH", pattern, "COUNT", "100");
    
    boost::redis::generic_response resp;
    auto status = redis_pool->Exec(req, resp);
    if (!status.ok()) break;
    
    const auto& nodes = resp.value();
    cursor = nodes[1].value;  // Next cursor
    for (size_t i = 3; i < nodes.size(); ++i) {
        if (nodes[i].depth == 2) {
            keys.push_back(nodes[i].value);
        }
    }
} while (cursor != "0");
```

### 2. Health Check Interference

**Issue**: Boost.redis auto-PING can interfere with custom command sequences.

**Workaround**: Disable health check interval:

```cpp
boost::redis::config cfg;
cfg.health_check_interval = std::chrono::seconds::zero();
```

## Recommendations

### For Development

1. **Use WSL Redis** for local development on Windows
   - Start Redis in WSL: `redis-server --port 5000`
   - Configure client to connect to `127.0.0.1:5000`
   - Avoids Windows Redis port issues entirely

2. **Keep Windows Redis for simple tests**
   - Bare connection tests
   - Small string payloads
   - Protocol validation

### For Production

1. **Deploy on Linux** (native or containerized)
   - Use official Redis Docker image or native Linux build
   - Avoid Windows Redis port for binary payload workloads

2. **Use SCAN instead of KEYS**
   - More efficient for large keyspaces
   - Avoids Boost.redis connection abort issue
   - Non-blocking iteration

3. **Disable health check if using custom command sequences**
   - Set `health_check_interval = std::chrono::seconds::zero()`
   - Prevents auto-PING interference

### For Testing

1. **E2E tests should target WSL Redis**
   - Config: `"redis": {"host": "127.0.0.1", "port": 5000}`
   - Ensures binary payload compatibility

2. **Unit tests can use Windows Redis**
   - Small payloads only
   - Protocol-level validation

3. **Add timeout to all Redis operations**
   - Use `Exec` overload with timeout parameter
   - Prevents indefinite hangs in case of issues

## Code Examples

### Correct SCAN Usage

```cpp
std::vector<std::string> ScanKeys(const std::string& pattern) {
    std::vector<std::string> keys;
    std::string cursor = "0";
    
    do {
        boost::redis::request req;
        req.push("SCAN", cursor, "MATCH", pattern, "COUNT", "100");
        
        boost::redis::generic_response resp;
        auto status = redis_pool_->Exec(req, resp);
        if (!status.ok()) {
            logger_.error("SCAN failed: {}", status.message());
            break;
        }
        
        const auto& nodes = resp.value();
        if (nodes.size() < 3) {
            logger_.warn("SCAN returned unexpected node count: {}", nodes.size());
            break;
        }
        
        cursor = nodes[1].value;
        for (size_t i = 3; i < nodes.size(); ++i) {
            if (nodes[i].depth == 2) {
                keys.push_back(nodes[i].value());
            }
        }
    } while (cursor != "0");
    
    return keys;
}
```

### Redis Pool Configuration

```cpp
boost::redis::config cfg;
cfg.addr.host = "127.0.0.1";
cfg.addr.port = "5000";  // WSL Redis
cfg.health_check_interval = std::chrono::seconds::zero();  // Disable auto-PING

auto conn = std::make_shared<boost::redis::connection>(io_context);
conn->async_run(cfg, {}, [](auto ec) {
    if (ec) {
        // Handle error
    }
});
```

### Safe LRANGE with Timeout

```cpp
boost::redis::request req;
req.push("LRANGE", key, "0", "-1");

boost::redis::response<std::vector<std::string>> resp;
auto status = redis_pool->Exec(req, resp, std::chrono::seconds(5));  // 5s timeout

if (!status.ok()) {
    logger_.error("LRANGE failed: {}", status.message());
    return status;
}

auto records = std::get<0>(resp).value();
// Process records...
```

## Related Files

- `src/semantic_cache/redis_connection_pool.h` — Redis pool implementation
- `src/semantic_cache/redis_connection_pool.cpp` — SCAN usage example
- `tools/redis_bare_connection_test.cpp` — Simple Redis connectivity test
- `tools/l3_compression_e2e_test.cpp` — E2E test with binary payloads
- `tools/l3_compression_e2e_test.json` — Config with WSL Redis port

## Timeline

- **2026-05-26**: Discovered Windows Redis `LRANGE` blocking issue during E2E testing
- **2026-05-26**: Confirmed WSL Redis workaround resolves the issue
- **2026-05-26**: Documented `KEYS` command abort and `SCAN` workaround
- **2026-05-27**: Added crash dump infrastructure to aid debugging
- **2026-05-27**: E2E test passes with WSL Redis, validates full L3 compression pipeline

## Future Work

1. **Investigate Windows Redis source** — determine exact cause of binary payload issue
2. **Report to Redis Windows maintainers** — if issue is reproducible with minimal example
3. **Consider alternative Redis clients** — if Boost.redis issues persist (e.g., redis-plus-plus, hiredis)
4. **Add Redis version detection** — warn if running on Windows Redis with binary payloads
