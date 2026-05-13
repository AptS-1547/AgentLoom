cmake_minimum_required(VERSION 3.20)
project(MultimodalInferenceService VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# vcpkg 集成
if(WIN32)
    set(BERT_VCPKG_TRIPLET_DEFAULT "x64-windows")
elseif(UNIX AND NOT APPLE)
    set(BERT_VCPKG_TRIPLET_DEFAULT "x64-linux")
else()
    set(BERT_VCPKG_TRIPLET_DEFAULT "")
endif()

set(BERT_VCPKG_TRIPLET "${BERT_VCPKG_TRIPLET_DEFAULT}" CACHE STRING "vcpkg target triplet for dependency lookup")

if(BERT_VCPKG_TRIPLET)
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${BERT_VCPKG_TRIPLET}")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${BERT_VCPKG_TRIPLET}/share")
endif()

# 查找依赖
if(BERT_VCPKG_TRIPLET)
    set(BERT_PROTOC_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${BERT_VCPKG_TRIPLET}/tools/protobuf/protoc")
    if(WIN32)
        string(APPEND BERT_PROTOC_EXECUTABLE ".exe")
    endif()
    if(EXISTS "${BERT_PROTOC_EXECUTABLE}")
        set(Protobuf_PROTOC_EXECUTABLE "${BERT_PROTOC_EXECUTABLE}")
    endif()
endif()
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# ONNX Runtime 预编译二进制
if(WIN32)
    set(BERT_DEFAULT_ONNXRUNTIME_CPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime-win-x64-1.17.1")
    set(BERT_DEFAULT_ONNXRUNTIME_GPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime-win-x64-gpu-1.20.1")
    set(BERT_DEFAULT_CUDA_RUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/cuda-runtime/bin")
    set(BERT_ONNXRUNTIME_SHARED_LIB_RELATIVE "lib/onnxruntime.dll")
    set(BERT_ONNXRUNTIME_IMPORT_LIB_RELATIVE "lib/onnxruntime.lib")
    set(BERT_ONNXRUNTIME_RUNTIME_GLOB "lib/onnxruntime*.dll")
elseif(UNIX AND NOT APPLE)
    set(BERT_DEFAULT_ONNXRUNTIME_CPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime-linux-x64-1.17.1")
    set(BERT_DEFAULT_ONNXRUNTIME_GPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime-linux-x64-gpu-1.20.1")
    set(BERT_DEFAULT_CUDA_RUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/cuda-runtime/lib64")
    set(BERT_ONNXRUNTIME_SHARED_LIB_RELATIVE "lib/libonnxruntime.so")
    set(BERT_ONNXRUNTIME_IMPORT_LIB_RELATIVE "")
    set(BERT_ONNXRUNTIME_RUNTIME_GLOB "lib/libonnxruntime*.so*")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

set(ONNXRUNTIME_CPU_ROOT "${BERT_DEFAULT_ONNXRUNTIME_CPU_ROOT}" CACHE PATH "Path to the CPU ONNX Runtime package")
set(ONNXRUNTIME_GPU_ROOT "${BERT_DEFAULT_ONNXRUNTIME_GPU_ROOT}" CACHE PATH "Path to the GPU ONNX Runtime package")
set(CUDA_RUNTIME_DLL_ROOT "${BERT_DEFAULT_CUDA_RUNTIME_ROOT}" CACHE PATH "Directory containing CUDA/cuDNN runtime libraries")
set(BERT_USE_ONNXRUNTIME_GPU "AUTO" CACHE STRING "Use the GPU ONNX Runtime package: AUTO, ON, or OFF")
set_property(CACHE BERT_USE_ONNXRUNTIME_GPU PROPERTY STRINGS AUTO ON OFF)

set(BERT_ONNXRUNTIME_CPU_LIB "${ONNXRUNTIME_CPU_ROOT}/${BERT_ONNXRUNTIME_SHARED_LIB_RELATIVE}")
set(BERT_ONNXRUNTIME_GPU_LIB "${ONNXRUNTIME_GPU_ROOT}/${BERT_ONNXRUNTIME_SHARED_LIB_RELATIVE}")

if(BERT_USE_ONNXRUNTIME_GPU STREQUAL "ON")
    if(NOT EXISTS "${BERT_ONNXRUNTIME_GPU_LIB}")
        message(FATAL_ERROR "BERT_USE_ONNXRUNTIME_GPU=ON but GPU package was not found at ${ONNXRUNTIME_GPU_ROOT}")
    endif()
    set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_GPU_ROOT}")
elseif(BERT_USE_ONNXRUNTIME_GPU STREQUAL "AUTO")
    if(EXISTS "${BERT_ONNXRUNTIME_GPU_LIB}")
        set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_GPU_ROOT}")
    else()
        set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_CPU_ROOT}")
    endif()
else()
    set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_CPU_ROOT}")
endif()

set(BERT_ONNXRUNTIME_SHARED_LIB "${ONNXRUNTIME_ROOT}/${BERT_ONNXRUNTIME_SHARED_LIB_RELATIVE}")
if(NOT EXISTS "${BERT_ONNXRUNTIME_SHARED_LIB}")
    message(FATAL_ERROR "ONNX Runtime package not found at ${ONNXRUNTIME_ROOT}")
endif()

message(STATUS "Using ONNX Runtime from ${ONNXRUNTIME_ROOT}")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${BERT_ONNXRUNTIME_SHARED_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

if(WIN32)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/${BERT_ONNXRUNTIME_IMPORT_LIB_RELATIVE}"
    )
endif()

# llama.cpp 预编译路径
set(LLAMA_CPP_ROOT "D:/Users/21405/source/repos/LocalLLMPredict/llama.cpp" CACHE PATH "Path to llama.cpp repository")
set(LLAMA_CPP_BUILD "${LLAMA_CPP_ROOT}/build")

# ==================== Proto 代码生成 ====================

set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

# BERT proto（保留，客户端工具仍需要）
set(BERT_PROTO_FILE ${PROTO_DIR}/bert_inference.proto)
set(BERT_PROTO_SRC ${GENERATED_DIR}/bert_inference.pb.cc)
set(BERT_PROTO_HDR ${GENERATED_DIR}/bert_inference.pb.h)
set(BERT_GRPC_SRC ${GENERATED_DIR}/bert_inference.grpc.pb.cc)
set(BERT_GRPC_HDR ${GENERATED_DIR}/bert_inference.grpc.pb.h)

add_custom_command(
    OUTPUT ${BERT_PROTO_SRC} ${BERT_PROTO_HDR} ${BERT_GRPC_SRC} ${BERT_GRPC_HDR}
    COMMAND protobuf::protoc
    ARGS --cpp_out=${GENERATED_DIR}
         --grpc_out=${GENERATED_DIR}
         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         -I ${PROTO_DIR}
         ${BERT_PROTO_FILE}
    DEPENDS ${BERT_PROTO_FILE}
    COMMENT "Generating BERT protobuf/gRPC code"
)

# Multimodal proto（统一协议）
set(MULTIMODAL_PROTO_FILE ${PROTO_DIR}/multimodal_inference.proto)
set(MULTIMODAL_PROTO_SRC ${GENERATED_DIR}/multimodal_inference.pb.cc)
set(MULTIMODAL_PROTO_HDR ${GENERATED_DIR}/multimodal_inference.pb.h)
set(MULTIMODAL_GRPC_SRC ${GENERATED_DIR}/multimodal_inference.grpc.pb.cc)
set(MULTIMODAL_GRPC_HDR ${GENERATED_DIR}/multimodal_inference.grpc.pb.h)

add_custom_command(
    OUTPUT ${MULTIMODAL_PROTO_SRC} ${MULTIMODAL_PROTO_HDR} ${MULTIMODAL_GRPC_SRC} ${MULTIMODAL_GRPC_HDR}
    COMMAND protobuf::protoc
    ARGS --cpp_out=${GENERATED_DIR}
         --grpc_out=${GENERATED_DIR}
         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         -I ${PROTO_DIR}
         ${MULTIMODAL_PROTO_FILE}
    DEPENDS ${MULTIMODAL_PROTO_FILE}
    COMMENT "Generating Multimodal protobuf/gRPC code"
)

add_custom_target(generate_proto ALL DEPENDS
    ${BERT_PROTO_SRC} ${BERT_PROTO_HDR} ${BERT_GRPC_SRC} ${BERT_GRPC_HDR}
    ${MULTIMODAL_PROTO_SRC} ${MULTIMODAL_PROTO_HDR} ${MULTIMODAL_GRPC_SRC} ${MULTIMODAL_GRPC_HDR}
)

# Proto 库
add_library(bert_proto STATIC ${BERT_PROTO_SRC} ${BERT_GRPC_SRC})
add_dependencies(bert_proto generate_proto)
target_include_directories(bert_proto PUBLIC ${GENERATED_DIR})
target_link_libraries(bert_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)

add_library(multimodal_proto STATIC ${MULTIMODAL_PROTO_SRC} ${MULTIMODAL_GRPC_SRC})
add_dependencies(multimodal_proto generate_proto)
target_include_directories(multimodal_proto PUBLIC ${GENERATED_DIR})
target_link_libraries(multimodal_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)

# ==================== 公共库 ====================

add_library(server_common STATIC
    src/common/server_common.cpp
    src/common/server_common.h
    src/common/logger.cpp
    src/common/logger.h
)

target_include_directories(server_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
)

target_link_libraries(server_common PUBLIC spdlog::spdlog)

# ==================== 统一多模态推理服务端 ====================

add_executable(multimodal_inference_server
    src/server/multimodal_inference_server.cpp
    src/server/request_validation.cpp
    src/server/request_validation.h
    src/server/server_config.cpp
    src/server/server_config.h
    src/server/server_options.h
    src/server/vector_cache.cpp
    src/server/vector_cache.h
    src/server/vlm_cache.cpp
    src/server/vlm_cache.h
    src/models/llama_runner.cpp
    src/models/llama_runner.h
    src/models/onnx_model.cpp
    src/models/onnx_model.h
)

target_include_directories(multimodal_inference_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/models
    ${CMAKE_CURRENT_SOURCE_DIR}/src/server
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${GENERATED_DIR}
    ${ONNXRUNTIME_ROOT}/include
    ${LLAMA_CPP_ROOT}/include
    ${LLAMA_CPP_ROOT}/tools/mtmd
    ${LLAMA_CPP_ROOT}/common
    ${LLAMA_CPP_ROOT}/ggml/include
)

target_link_libraries(multimodal_inference_server PRIVATE
    multimodal_proto
    server_common
    onnxruntime
    gRPC::grpc++
    spdlog::spdlog
    ${LLAMA_CPP_BUILD}/src/Release/llama.lib
    ${LLAMA_CPP_BUILD}/tools/mtmd/Release/mtmd.lib
    ${LLAMA_CPP_BUILD}/common/Release/llama-common.lib
    ${LLAMA_CPP_BUILD}/ggml/src/Release/ggml.lib
    ${LLAMA_CPP_BUILD}/ggml/src/Release/ggml-base.lib
)

# 拷贝 llama.cpp DLL 到输出目录
add_custom_command(TARGET multimodal_inference_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LLAMA_CPP_BUILD}/bin/Release/llama.dll"
        "${LLAMA_CPP_BUILD}/bin/Release/ggml.dll"
        "${LLAMA_CPP_BUILD}/bin/Release/ggml-base.dll"
        "${LLAMA_CPP_BUILD}/bin/Release/ggml-cpu.dll"
        "${LLAMA_CPP_BUILD}/bin/Release/ggml-cuda.dll"
        "${LLAMA_CPP_BUILD}/bin/Release/mtmd.dll"
        "${LLAMA_CPP_BUILD}/bin/Release/llama-common.dll"
        $<TARGET_FILE_DIR:multimodal_inference_server>
    COMMENT "Copying llama.cpp DLLs"
)

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

# ==================== 平台适配 ====================

if(UNIX AND NOT APPLE)
    foreach(target multimodal_inference_server bert_inference_client bert_benchmark_client)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
        )
    endforeach()
endif()

# 安装
install(TARGETS multimodal_inference_server bert_inference_client bert_benchmark_client RUNTIME DESTINATION bin)

# 复制 ONNX Runtime 运行时库
file(GLOB ONNXRUNTIME_RUNTIME_LIBS "${ONNXRUNTIME_ROOT}/${BERT_ONNXRUNTIME_RUNTIME_GLOB}")
foreach(ONNXRUNTIME_RUNTIME_LIB ${ONNXRUNTIME_RUNTIME_LIBS})
    add_custom_command(TARGET multimodal_inference_server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_RUNTIME_LIB}"
            $<TARGET_FILE_DIR:multimodal_inference_server>
    )
endforeach()

# 复制 vcpkg 动态库依赖
if(WIN32 AND BERT_VCPKG_TRIPLET)
    file(GLOB VCPKG_RUNTIME_DLLS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${BERT_VCPKG_TRIPLET}/bin/*.dll")
    foreach(target multimodal_inference_server bert_inference_client bert_benchmark_client)
        foreach(VCPKG_RUNTIME_DLL ${VCPKG_RUNTIME_DLLS})
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VCPKG_RUNTIME_DLL}"
                    $<TARGET_FILE_DIR:${target}>
            )
        endforeach()
    endforeach()
endif()

if(UNIX AND NOT APPLE AND BERT_VCPKG_TRIPLET)
    file(GLOB VCPKG_RUNTIME_SOS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${BERT_VCPKG_TRIPLET}/lib/*.so*")
    foreach(target multimodal_inference_server bert_inference_client bert_benchmark_client)
        foreach(VCPKG_RUNTIME_SO ${VCPKG_RUNTIME_SOS})
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VCPKG_RUNTIME_SO}"
                    $<TARGET_FILE_DIR:${target}>
            )
        endforeach()
    endforeach()
endif()

# 复制 CUDA / cuDNN 运行时库（如果存在）
if(EXISTS "${CUDA_RUNTIME_DLL_ROOT}")
    if(WIN32)
        file(GLOB CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_DLL_ROOT}/*.dll")
    elseif(UNIX AND NOT APPLE)
        file(GLOB CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_DLL_ROOT}/*.so*")
    endif()
    foreach(CUDA_RUNTIME_LIB ${CUDA_RUNTIME_LIBS})
        add_custom_command(TARGET multimodal_inference_server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CUDA_RUNTIME_LIB}"
                $<TARGET_FILE_DIR:multimodal_inference_server>
        )
    endforeach()
endif()
