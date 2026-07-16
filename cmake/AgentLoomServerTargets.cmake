# ==================== 推理服务端 ====================

add_library(agent_emotion_server STATIC
    src/server/grpc/grpc_error.cpp
    src/server/grpc/grpc_error.h
    src/server/grpc/emotion_grpc_service.cpp
    src/server/grpc/emotion_grpc_service.h
    src/service/inference/emotion_inference_service.cpp
    src/service/inference/emotion_inference_service.h
    src/service/inference/request_validation.cpp
    src/service/inference/request_validation.h
)

target_include_directories(agent_emotion_server PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/server/grpc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/inference
)

if(UNIX AND NOT APPLE)
    target_include_directories(agent_emotion_server PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/config
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
endif()

target_link_libraries(agent_emotion_server PUBLIC
    multimodal_proto
    agent_core
    agent_bert_models
    server_runtime
    gRPC::grpc++
    spdlog::spdlog
)
if(WIN32)
    target_link_libraries(agent_emotion_server PUBLIC agent_config)
else()
    target_link_libraries(agent_emotion_server PUBLIC agent_net)
endif()

add_library(agent_server STATIC
    src/server/grpc/grpc_error.cpp
    src/server/grpc/grpc_error.h
    src/server/grpc/multimodal_grpc_service.cpp
    src/server/grpc/multimodal_grpc_service.h
    src/service/inference/multimodal_service.cpp
    src/service/inference/multimodal_service.h
    src/service/inference/request_validation.cpp
    src/service/inference/request_validation.h
)

target_include_directories(agent_server PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/server/grpc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/inference
)

target_link_libraries(agent_server PUBLIC
    multimodal_proto
    agent_core
    agent_ipc
    agent_media_inference
    agent_service
    agent_models
    agent_cache
    agent_semantic_cache
    agent_vector
    server_runtime
    agent_config
    gRPC::grpc++
    spdlog::spdlog
)

if(BERT_BUILD_EMOTION_INFERENCE_SERVER)
    add_executable(emotion_inference_server
        src/server/main/emotion_inference_server.cpp
    )

    target_link_libraries(emotion_inference_server PRIVATE agent_emotion_server)
    link_whole_archive(emotion_inference_server agent_config)
endif()

if(BERT_BUILD_MULTIMODAL_INFERENCE_SERVER)
    add_executable(multimodal_inference_server
        src/server/main/multimodal_inference_server.cpp
    )

    target_link_libraries(multimodal_inference_server PRIVATE agent_server)
    if(WIN32)
        link_whole_archive(multimodal_inference_server agent_config)
    endif()

    copy_runtime_files(multimodal_inference_server ${LLAMA_CPP_RUNTIME_FILES})
endif()

if(BERT_BUILD_TESTS)
    add_executable(inference_grpc_tests
        tests/server/inference_grpc_service_test.cpp
        src/server/grpc/grpc_error.cpp
        src/server/grpc/emotion_grpc_service.cpp
        src/server/grpc/multimodal_grpc_service.cpp
        src/service/inference/request_validation.cpp
    )

    target_include_directories(inference_grpc_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/server/grpc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/service/inference
    )

    target_link_libraries(inference_grpc_tests PRIVATE
        multimodal_proto
        agent_core
        agent_ipc_grpc
        server_runtime
        agent_config
        gRPC::grpc++
        spdlog::spdlog
        GTest::gtest_main
    )

    copy_runtime_files(inference_grpc_tests ${VCPKG_RUNTIME_DLLS})
    gtest_discover_tests(inference_grpc_tests DISCOVERY_MODE PRE_TEST)
endif()

# ==================== 客户端工具（BERT 协议，用于向后兼容测试） ====================

add_executable(bert_inference_client
    src/client/client_test.cpp
)

target_include_directories(bert_inference_client PRIVATE ${GENERATED_DIR})
target_link_libraries(bert_inference_client PRIVATE bert_proto gRPC::grpc++)

add_executable(bert_benchmark_client
    src/client/benchmark_client.cpp
)

target_include_directories(bert_benchmark_client PRIVATE ${GENERATED_DIR})
target_link_libraries(bert_benchmark_client PRIVATE bert_proto gRPC::grpc++)
