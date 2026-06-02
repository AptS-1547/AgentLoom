// redis++ connection smoke test
// Goal: verify redis++ library can perform basic operations against local Redis.
//
// Run: .\build\x64-Release-Tests\Release\redis_bare_connection_test.exe

#include <sw/redis++/redis++.h>
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    using namespace std::chrono_literals;

    std::cout << "[bare] Starting redis++ connection test...\n";

    try {
        sw::redis::ConnectionOptions opts;
        opts.host = "127.0.0.1";
        opts.port = 5000;
        opts.socket_timeout = std::chrono::milliseconds(2000);

        std::cout << "[bare] Connecting to " << opts.host << ":" << opts.port << "...\n";

        sw::redis::Redis redis(opts);

        std::cout << "[bare] Connection established\n";

        // 1. PING
        {
            redis.ping();
            std::cout << "[bare] PING -> PONG\n";
        }

        const std::string key = "bare:test:key";

        // 2. DEL key (cleanup any leftover)
        {
            auto deleted = redis.del(key);
            std::cout << "[bare] DEL -> " << deleted << "\n";
        }

        // 3. RPUSH 3 values
        for (int i = 0; i < 3; ++i) {
            auto len = redis.rpush(key, "value-" + std::to_string(i));
            std::cout << "[bare] RPUSH " << i << " -> len=" << len << "\n";
        }

        // 4. LRANGE
        {
            std::vector<std::string> vals;
            redis.lrange(key, 0, -1, std::back_inserter(vals));
            std::cout << "[bare] LRANGE -> " << vals.size() << " items\n";
            for (const auto& v : vals) {
                std::cout << "  - " << v << "\n";
            }
        }

        // 5. EXPIRE
        {
            auto result = redis.expire(key, std::chrono::seconds(86400));
            std::cout << "[bare] EXPIRE -> " << result << "\n";
        }

        // 5.5. PING again
        {
            redis.ping();
            std::cout << "[bare] PING-2 -> PONG\n";
        }

        // 6. SCAN
        {
            std::vector<std::string> matched_keys;
            long long cursor = 0;
            do {
                cursor = redis.scan(cursor, "bare:test:*", 100, std::back_inserter(matched_keys));
            } while (cursor != 0);
            std::cout << "[bare] SCAN -> " << matched_keys.size() << " keys matched\n";
            for (const auto& k : matched_keys) {
                std::cout << "  + " << k << "\n";
            }
        }

        // 7. LRANGE again
        {
            std::vector<std::string> vals;
            redis.lrange(key, 0, -1, std::back_inserter(vals));
            std::cout << "[bare] LRANGE-2 -> " << vals.size() << " items\n";
        }

        // 8. HSET/HGETALL test
        {
            const std::string hkey = "bare:test:hash";
            redis.del(hkey);

            redis.hset(hkey, "field1", "value1");
            redis.hset(hkey, "field2", "value2");
            redis.hset(hkey, "field3", "value3");
            std::cout << "[bare] HSET x3 -> OK\n";

            std::unordered_map<std::string, std::string> result;
            redis.hgetall(hkey, std::inserter(result, result.begin()));
            std::cout << "[bare] HGETALL -> " << result.size() << " fields\n";
            for (const auto& [field, value] : result) {
                std::cout << "  - " << field << " = " << value << "\n";
            }

            redis.del(hkey);
        }

        // 9. Cleanup
        {
            redis.del(key);
            std::cout << "[bare] cleanup done\n";
        }

        std::cout << "[bare] all tests passed\n";
        return 0;

    } catch (const sw::redis::Error& e) {
        std::cerr << "[bare] redis++ error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[bare] exception: " << e.what() << "\n";
        return 1;
    }
}
