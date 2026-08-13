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
