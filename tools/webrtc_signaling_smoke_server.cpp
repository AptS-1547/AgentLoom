#include "http_server.h"
#include "memory_pool.h"
#include "opencv_frame_sampler.h"
#include "redis_connection_pool.h"
#include "runtime_maintenance_service.h"
#include "thread_pool.h"
#include "webrtc_media_pipeline.h"
#include "webrtc_session_registry.h"
#include "webrtc_signaling_handler.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic_bool g_stop_requested{false};

void OnSignal(int) {
    g_stop_requested.store(true, std::memory_order_release);
}

unsigned short ParsePort(const std::string& value) {
    const auto number = std::stoul(value);
    if (number == 0 || number > 65535) {
        throw std::out_of_range("port must be in range 1..65535");
    }
    return static_cast<unsigned short>(number);
}

net::BeastHttpResponse TextResponse(net::http::status status, std::string body) {
    net::BeastHttpResponse response{status, 11};
    response.set(net::http::field::content_type, "text/plain; charset=utf-8");
    response.body() = std::move(body);
    response.prepare_payload();
    return response;
}

net::BeastHttpResponse JsonResponse(net::http::status status, const nlohmann::json& body) {
    net::BeastHttpResponse response{status, 11};
    response.set(net::http::field::content_type, "application/json; charset=utf-8");
    response.body() = body.dump();
    response.prepare_payload();
    return response;
}

class DiskFrameSampler final : public media::IFrameSampler {
public:
    explicit DiskFrameSampler(std::filesystem::path frame_dir)
        : frame_dir_(std::move(frame_dir)) {
        std::filesystem::create_directories(frame_dir_);
    }

    core::Result<media::FrameSamplingDecision> Evaluate(const media::VideoFrameView& frame) override {
        if (frame.format != media::VideoPixelFormat::Rgb || frame.width == 0 || frame.height == 0 || frame.bytes.empty()) {
            return media::FrameSamplingDecision{false, 0.0, "unsupported-frame"};
        }

        const auto expected = static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.height) * 3;
        if (frame.bytes.size() < expected) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "RGB frame buffer is smaller than width*height*3");
        }

        const auto index = next_index_.fetch_add(1, std::memory_order_relaxed);
        const auto path = frame_dir_ / (SafeName(frame.session_id) + "_frame_" + std::to_string(index) + ".ppm");
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file) {
            return core::Status::Error(core::ErrorCode::InternalError, "failed to open frame output file");
        }
        file << "P6\n" << frame.width << " " << frame.height << "\n255\n";
        file.write(frame.bytes.data(), static_cast<std::streamsize>(expected));
        if (!file) {
            return core::Status::Error(core::ErrorCode::InternalError, "failed to write frame output file");
        }
        return media::FrameSamplingDecision{false, 1.0, "written"};
    }

private:
    static std::string SafeName(const std::string& value) {
        std::string out;
        out.reserve(value.size());
        for (char ch : value) {
            const auto safe = (ch >= 'a' && ch <= 'z') ||
                              (ch >= 'A' && ch <= 'Z') ||
                              (ch >= '0' && ch <= '9') ||
                              ch == '-' ||
                              ch == '_';
            out.push_back(safe ? ch : '_');
        }
        return out.empty() ? "session" : out;
    }

    std::filesystem::path frame_dir_;
    std::atomic<std::uint64_t> next_index_{0};
};

class GatedDiskFrameSampler final : public media::IFrameSampler {
public:
    GatedDiskFrameSampler(std::shared_ptr<media::IFrameSampler> gate,
                          std::shared_ptr<media::IFrameSampler> sink)
        : gate_(std::move(gate)), sink_(std::move(sink)) {}

    core::Result<media::FrameSamplingDecision> Evaluate(const media::VideoFrameView& frame) override {
        auto decision = gate_->Evaluate(frame);
        if (!decision.ok()) {
            return decision.status();
        }
        if (!decision.value().submit_to_vlm) {
            return decision;
        }
        auto sink_decision = sink_->Evaluate(frame);
        if (!sink_decision.ok()) {
            return sink_decision.status();
        }
        return decision;
    }

private:
    std::shared_ptr<media::IFrameSampler> gate_;
    std::shared_ptr<media::IFrameSampler> sink_;
};

} // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, OnSignal);
    std::signal(SIGTERM, OnSignal);

    std::string address = "127.0.0.1";
    unsigned short port = 18180;
    std::size_t io_threads = 2;
    std::filesystem::path frame_dir = "logs/webrtc_frames";
    std::string sampler_mode = "all";
    std::string redis_host = "127.0.0.1";
    std::string redis_port = "6379";
    std::string redis_password;
    std::string redis_key_prefix = "rtc:session:";
    std::vector<media::WebRtcIceServerConfig> ice_servers;
    bool redis_checkpoint = false;
    bool adaptive_sampler = false;
    media::VideoDecoderPreference decoder = media::VideoDecoderPreference::Auto;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--address" || arg == "-a") && i + 1 < argc) {
            address = argv[++i];
        } else if (arg.rfind("--address=", 0) == 0) {
            address = arg.substr(10);
        } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            port = ParsePort(argv[++i]);
        } else if (arg.rfind("--port=", 0) == 0) {
            port = ParsePort(arg.substr(7));
        } else if ((arg == "--io-threads" || arg == "-t") && i + 1 < argc) {
            io_threads = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg.rfind("--io-threads=", 0) == 0) {
            io_threads = static_cast<std::size_t>(std::stoul(arg.substr(13)));
        } else if (arg == "--decoder" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "software") {
                decoder = media::VideoDecoderPreference::Software;
            } else if (value == "nvidia") {
                decoder = media::VideoDecoderPreference::Nvidia;
            } else if (value == "vaapi") {
                decoder = media::VideoDecoderPreference::Vaapi;
            } else {
                decoder = media::VideoDecoderPreference::Auto;
            }
        } else if (arg == "--frame-dir" && i + 1 < argc) {
            frame_dir = argv[++i];
        } else if (arg == "--sampler" && i + 1 < argc) {
            sampler_mode = argv[++i];
        } else if (arg == "--adaptive-sampler") {
            adaptive_sampler = true;
        } else if (arg == "--redis-checkpoint") {
            redis_checkpoint = true;
        } else if (arg == "--redis-host" && i + 1 < argc) {
            redis_host = argv[++i];
            redis_checkpoint = true;
        } else if (arg.rfind("--redis-host=", 0) == 0) {
            redis_host = arg.substr(13);
            redis_checkpoint = true;
        } else if (arg == "--redis-port" && i + 1 < argc) {
            redis_port = argv[++i];
            redis_checkpoint = true;
        } else if (arg.rfind("--redis-port=", 0) == 0) {
            redis_port = arg.substr(13);
            redis_checkpoint = true;
        } else if (arg == "--redis-password" && i + 1 < argc) {
            redis_password = argv[++i];
            redis_checkpoint = true;
        } else if (arg.rfind("--redis-password=", 0) == 0) {
            redis_password = arg.substr(17);
            redis_checkpoint = true;
        } else if (arg == "--redis-key-prefix" && i + 1 < argc) {
            redis_key_prefix = argv[++i];
            redis_checkpoint = true;
        } else if (arg.rfind("--redis-key-prefix=", 0) == 0) {
            redis_key_prefix = arg.substr(19);
            redis_checkpoint = true;
        } else if (arg == "--ice-server" && i + 1 < argc) {
            ice_servers.push_back(media::WebRtcIceServerConfig{.urls = {argv[++i]}});
        } else if (arg.rfind("--ice-server=", 0) == 0) {
            ice_servers.push_back(media::WebRtcIceServerConfig{.urls = {arg.substr(13)}});
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: webrtc_signaling_smoke_server [--address ADDR] [--port PORT] "
                         "[--io-threads N] [--decoder auto|software|nvidia|vaapi] "
                         "[--frame-dir DIR] [--sampler all|opencv] [--adaptive-sampler] "
                         "[--redis-checkpoint] [--redis-host HOST] [--redis-port PORT] "
                         "[--redis-password PASSWORD] [--redis-key-prefix PREFIX] "
                         "[--ice-server URL]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return 1;
        }
    }

    auto compute_pool = std::make_shared<core::ThreadPool>(core::ThreadPoolOptions{
        .worker_count = 2,
        .queue_capacity = 64,
        .name = "webrtc-smoke-compute",
    });
    auto pool_status = compute_pool->Start();
    if (!pool_status.ok()) {
        std::cerr << "failed to start compute pool: " << pool_status.message() << "\n";
        return 1;
    }

    auto disk_sampler = std::make_shared<DiskFrameSampler>(frame_dir);
    std::shared_ptr<media::IFrameSampler> frame_sampler = disk_sampler;
    if (sampler_mode == "opencv") {
        media::OpenCvFrameSamplerConfig sampler_config;
        sampler_config.peak_threshold = 0.25;
        sampler_config.cooldown_seconds = 0.5;
        sampler_config.temporal_vote_required = 1;
        sampler_config.min_component_area_ratio = 0.001;
        sampler_config.adaptive_enabled = adaptive_sampler;
        frame_sampler = std::make_shared<GatedDiskFrameSampler>(
            std::make_shared<media::OpenCvFrameSampler>(sampler_config),
            disk_sampler);
    }

    media::WebRtcSignalingOptions signaling_options{
        .max_sessions = 16,
        .decode_options = {.video_decoder = decoder},
        .compute_pool = compute_pool,
        .frame_sampler = frame_sampler,
        .ice_servers = ice_servers,
    };

    std::shared_ptr<agent::semantic_cache::RedisConnectionPool> redis_pool;
    std::shared_ptr<media::IWebRtcCheckpointStore> checkpoint_store;
    if (redis_checkpoint) {
        agent::semantic_cache::RedisPoolOptions redis_options;
        redis_options.host = redis_host;
        redis_options.port = redis_port;
        redis_options.password = redis_password;
        redis_options.pool_size = 4;
        redis_pool = std::make_shared<agent::semantic_cache::RedisConnectionPool>(redis_options);
        auto redis_start = redis_pool->Start();
        if (!redis_start.ok()) {
            std::cerr << "failed to start redis checkpoint pool: " << redis_start.message() << "\n";
            compute_pool->Shutdown();
            return 1;
        }
        checkpoint_store = std::make_shared<media::RedisWebRtcCheckpointStore>(redis_pool, redis_key_prefix);
        std::cout << "webrtc checkpoint redis enabled at " << redis_host << ":" << redis_port
                  << " prefix=" << redis_key_prefix << "\n";
    }

    auto registry_ref = std::make_shared<std::weak_ptr<media::WebRtcSessionRegistry>>();
    signaling_options.frame_observer = [registry_ref](const media::VideoFrameView& frame) {
        if (auto registry = registry_ref->lock()) {
            static_cast<void>(registry->RecordFrame(frame.session_id, frame.frame_id));
        }
    };

    media::WebRtcPeerFactory peer_factory =
        [signaling_options, registry_ref](const std::string& session_id,
                                          const std::string& name) -> core::Result<std::shared_ptr<media::IWebRtcPeer>> {
        auto peer = media::WebRtcMediaPipeline::Create(media::WebRtcMediaPipelineOptions{
            .session_id = session_id,
            .name = name,
            .decode = signaling_options.decode_options,
            .compute_pool = signaling_options.compute_pool,
            .io_pool = signaling_options.io_pool,
            .frame_sampler = signaling_options.frame_sampler,
            .vlm_client = signaling_options.vlm_client,
            .frame_observer = signaling_options.frame_observer,
            .vision_event_sink = signaling_options.vision_event_sink,
            .status_observer = [registry_ref](std::string failed_session_id, core::Status status, bool fatal) {
                if (!fatal) {
                    return;
                }
                if (auto registry = registry_ref->lock()) {
                    static_cast<void>(registry->MarkFailed(failed_session_id, status.message()));
                }
            },
        });
        if (!peer.ok()) {
            return peer.status();
        }
        return std::static_pointer_cast<media::IWebRtcPeer>(std::move(peer).value());
    };
    auto registry = std::make_shared<media::WebRtcSessionRegistry>(
        media::WebRtcSessionRegistryOptions{
            .max_sessions = signaling_options.max_sessions,
            .checkpoint_ttl = std::chrono::seconds(300),
            .signaling_timeout = std::chrono::seconds(30),
            .reconnect_grace = std::chrono::seconds(30),
        },
        std::move(peer_factory),
        checkpoint_store);
    *registry_ref = registry;
    media::WebRtcSignalingHandler signaling(signaling_options, registry);

    agent::service::gateway::RuntimeMaintenanceService maintenance;
    auto maintenance_status = maintenance.RegisterTask(
        std::make_shared<agent::service::gateway::WebRtcSessionMaintenanceTask>(
            registry,
            std::chrono::seconds(5)));
    if (!maintenance_status.ok()) {
        std::cerr << "failed to register webrtc maintenance task: " << maintenance_status.message() << "\n";
        if (redis_pool) {
            redis_pool->Shutdown();
        }
        compute_pool->Shutdown();
        return 1;
    }

    net::HttpServer server({.address = address, .port = port, .io_threads = io_threads});
    server.SetHttpRequestHandler([registry](std::shared_ptr<net::IHttpRequest> request) {
        const auto target = request->message().target();
        if (target == "/" || target == "/health") {
            request->Respond(net::http::message_generator(TextResponse(net::http::status::ok, "ok")));
            return;
        }
        if (target == "/debug/vision/sessions") {
            auto snapshots = registry->SnapshotCheckpoints();
            nlohmann::json sessions = nlohmann::json::array();
            for (const auto& checkpoint : snapshots) {
                sessions.push_back({
                    {"session_id", checkpoint.session_id},
                    {"trace_id", checkpoint.trace_id},
                    {"state", checkpoint.state},
                    {"connection_id", checkpoint.connection_id},
                    {"last_frame_id", checkpoint.last_frame_id},
                    {"updated_at_ms", checkpoint.updated_at_ms},
                    {"last_signaling_at_ms", checkpoint.last_signaling_at_ms},
                    {"last_ice_at_ms", checkpoint.last_ice_at_ms},
                    {"last_frame_at_ms", checkpoint.last_frame_at_ms},
                    {"has_reconnect_token", !checkpoint.reconnect_token.empty()},
                });
            }
            request->Respond(net::http::message_generator(JsonResponse(
                net::http::status::ok,
                {
                    {"session_count", snapshots.size()},
                    {"sessions", std::move(sessions)},
                })));
            return;
        }
        request->Respond(net::http::message_generator(TextResponse(net::http::status::not_found, "not found")));
    });
    server.SetWebSocketStreamHandler("/ws/vision/signaling", [&signaling](std::shared_ptr<net::IWebSocketStreamRequest> request) {
        signaling.Handle(std::move(request));
    });
    server.SetWebSocketCloseHandler([&signaling](const net::ConnectionCloseInfo& close_info) {
        if (close_info.connection_id != 0) {
            signaling.RemoveConnection(close_info.connection_id);
        }
    });

    auto start = server.Start();
    if (!start.ok()) {
        std::cerr << "failed to start server: " << start.message() << "\n";
        if (redis_pool) {
            redis_pool->Shutdown();
        }
        compute_pool->Shutdown();
        return 1;
    }
    auto maintenance_start = maintenance.Start();
    if (!maintenance_start.ok()) {
        std::cerr << "failed to start maintenance service: " << maintenance_start.message() << "\n";
        server.Stop();
        if (redis_pool) {
            redis_pool->Shutdown();
        }
        compute_pool->Shutdown();
        return 1;
    }

    std::cout << "webrtc signaling smoke server listening on http://" << address << ":" << server.port() << "\n";
    while (!g_stop_requested.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    server.Stop();
    maintenance.Stop();
    if (redis_pool) {
        redis_pool->Shutdown();
    }
    compute_pool->Shutdown();
    return 0;
}
