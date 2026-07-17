add_library(agent_ipc STATIC
    src/ipc/inference_frame_ipc_control.cpp
    src/ipc/inference_frame_ipc_control.h
    src/ipc/inference_frame_ipc_lifecycle.cpp
    src/ipc/inference_frame_ipc_lifecycle.h
    src/ipc/inference_frame_shared_memory.cpp
    src/ipc/inference_frame_shared_memory.h
)

target_include_directories(agent_ipc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_link_libraries(agent_ipc PUBLIC
    agent_core
    boost_interprocess_headers
    spdlog::spdlog
)

add_library(agent_ipc_grpc STATIC
    src/ipc/inference_frame_ipc_grpc_signal.cpp
    src/ipc/inference_frame_ipc_grpc_signal.h
)

target_include_directories(agent_ipc_grpc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc
    ${GENERATED_DIR}
)

target_link_libraries(agent_ipc_grpc PUBLIC
    agent_ipc
    multimodal_proto
    gRPC::grpc++
)

add_library(agent_media_inference STATIC
    src/media/inference_frame_backlog.cpp
    src/media/inference_frame_backlog.h
    src/media/inference_frame_coordinator.cpp
    src/media/inference_frame_coordinator.h
    src/media/inference_frame_ipc_receiver.cpp
    src/media/inference_frame_ipc_receiver.h
    src/media/ordered_inference_frame_admission.cpp
    src/media/ordered_inference_frame_admission.h
    src/media/inference_frame_spool.cpp
    src/media/inference_frame_spool.h
    src/media/inference_frame_spool_replayer.cpp
    src/media/inference_frame_spool_replayer.h
    src/media/vision_inference_interfaces.h
)

target_include_directories(agent_media_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/media
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_link_libraries(agent_media_inference PUBLIC
    agent_core
    agent_ipc
    boost_interprocess_headers
    spdlog::spdlog
)

add_library(agent_media_vlm_grpc STATIC
    src/media/grpc_vlm_vision_client.cpp
    src/media/grpc_vlm_vision_client.h
)

target_include_directories(agent_media_vlm_grpc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/media
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_include_directories(agent_media_vlm_grpc PRIVATE
    ${GENERATED_DIR}
)

target_link_libraries(agent_media_vlm_grpc PUBLIC
    agent_media_inference
)

target_link_libraries(agent_media_vlm_grpc PRIVATE
    multimodal_proto
    gRPC::grpc++
    nlohmann_json::nlohmann_json
)

if(TARGET gstreamer_core)
    add_library(agent_media STATIC
        src/media/frame_encoding.cpp
        src/media/frame_encoding.h
        src/media/inference_frame_gateway_producer.cpp
        src/media/inference_frame_gateway_producer.h
        src/media/opencv_frame_sampler.cpp
        src/media/opencv_frame_sampler.h
        src/media/ordered_encoded_frame_sink.cpp
        src/media/ordered_encoded_frame_sink.h
        src/media/vision_runtime_interfaces.cpp
        src/media/vision_runtime_interfaces.h
        src/media/vision_inference_interfaces.h
        src/media/webrtc_media_pipeline.cpp
        src/media/webrtc_media_pipeline.h
        src/media/webrtc_session_registry.cpp
        src/media/webrtc_session_registry.h
        src/media/webrtc_signaling_handler.cpp
        src/media/webrtc_signaling_handler.h
        src/media/webrtc_bin.cpp
        src/media/webrtc_bin.h
    )

    target_include_directories(agent_media PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/media
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )

    target_link_libraries(agent_media PUBLIC
        agent_core
        agent_ipc
        agent_net
        agent_semantic_cache
        agent_vector
        ${OpenCV_LIBS}
        agentloom_gstreamer_dependency
    )

    target_compile_definitions(agent_media PRIVATE
        GST_USE_UNSTABLE_API
    )

    add_executable(media_gstreamer_probe
        src/media/gstreamer_probe.cpp
    )

    target_link_libraries(media_gstreamer_probe PRIVATE
        gstreamer_core
    )

    if(WIN32)
        set_target_properties(media_gstreamer_probe PROPERTIES
            VS_DEBUGGER_ENVIRONMENT "PATH=${GSTREAMER_ROOT}/bin;%PATH%;GST_PLUGIN_PATH=${GSTREAMER_ROOT}/lib/gstreamer-1.0"
        )
    endif()
endif()
