#include "http_server.h"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace {

std::atomic_bool g_stop_requested{false};

void OnSignal(int) {
    g_stop_requested.store(true, std::memory_order_release);
}

std::filesystem::path DefaultStaticRoot(const char* argv0) {
    std::filesystem::path exe_dir;
    if (argv0 && *argv0) {
        exe_dir = std::filesystem::absolute(argv0).parent_path();
    } else {
        exe_dir = std::filesystem::current_path();
    }

    const auto beside_exe = exe_dir / "static_site";
    if (std::filesystem::exists(beside_exe / "index.html")) {
        return beside_exe;
    }

    const auto source_tree = std::filesystem::current_path() / "tests" / "net" / "static_site";
    if (std::filesystem::exists(source_tree / "index.html")) {
        return source_tree;
    }

    return beside_exe;
}

unsigned short ParsePort(const std::string& value) {
    const auto number = std::stoul(value);
    if (number == 0 || number > 65535) {
        throw std::out_of_range("port must be in range 1..65535");
    }
    return static_cast<unsigned short>(number);
}

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, OnSignal);
    std::signal(SIGTERM, OnSignal);

    std::filesystem::path root = DefaultStaticRoot(argc > 0 ? argv[0] : nullptr);
    std::string address = "127.0.0.1";
    unsigned short port = 18080;
    std::size_t io_threads = 2;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--root" || arg == "-r") && i + 1 < argc) {
            root = argv[++i];
        } else if (arg.rfind("--root=", 0) == 0) {
            root = arg.substr(7);
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            port = ParsePort(argv[++i]);
        } else if (arg.rfind("--port=", 0) == 0) {
            port = ParsePort(arg.substr(7));
        } else if ((arg == "--address" || arg == "-a") && i + 1 < argc) {
            address = argv[++i];
        } else if (arg.rfind("--address=", 0) == 0) {
            address = arg.substr(10);
        } else if ((arg == "--io-threads" || arg == "-t") && i + 1 < argc) {
            io_threads = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg.rfind("--io-threads=", 0) == 0) {
            io_threads = static_cast<std::size_t>(std::stoul(arg.substr(13)));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: static_site_test_server [--root DIR] [--address ADDR] [--port PORT] [--io-threads N]\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 2;
        }
    }

    root = std::filesystem::absolute(root);
    if (!std::filesystem::exists(root / "index.html")) {
        std::cerr << "Static site index not found: " << (root / "index.html").string() << "\n";
        return 2;
    }

    net::HttpServerOptions options;
    options.address = address;
    options.port = port;
    options.io_threads = io_threads;
    options.static_files = net::StaticFileOptions{root, "index.html", true};

    net::HttpServer server(options);
    auto status = server.Start();
    if (!status.ok()) {
        std::cerr << "Failed to start static site test server: " << status.message() << "\n";
        return 1;
    }

    std::cout << "Static site test server running\n";
    std::cout << "  root: " << root.string() << "\n";
    std::cout << "  io_threads: " << io_threads << "\n";
    std::cout << "  url : http://" << address << ":" << server.port() << "/\n";
    std::cout << "Press Ctrl+C to stop.\n";

    while (!g_stop_requested.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    server.Stop();
    std::cout << "Static site test server stopped\n";
    return 0;
}
