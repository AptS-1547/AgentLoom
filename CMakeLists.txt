cmake_minimum_required(VERSION 3.20)
project(AgentLoom VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BERT_BUILD_TESTS "Build C++ unit tests" OFF)

if(MSVC)
    add_compile_options(/utf-8)
endif()

if(BERT_BUILD_TESTS)
    enable_testing()
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
find_package(OpenSSL CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(hiredis CONFIG REQUIRED)
find_package(redis++ CONFIG REQUIRED)
find_package(libzip CONFIG REQUIRED)
find_package(pugixml CONFIG REQUIRED)

if(WIN32 AND BERT_VCPKG_TRIPLET)
    file(GLOB VCPKG_RUNTIME_DLLS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${BERT_VCPKG_TRIPLET}/bin/*.dll")
endif()

function(require_path path label)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "${label} not found: ${path}")
    endif()
endfunction()

function(copy_runtime_files target)
    foreach(runtime_file IN LISTS ARGN)
        if(EXISTS "${runtime_file}")
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${runtime_file}"
                    $<TARGET_FILE_DIR:${target}>
            )
        endif()
    endforeach()
endfunction()

function(link_whole_archive target library)
    if(MSVC)
        target_link_options(${target} PRIVATE "/WHOLEARCHIVE:$<TARGET_FILE:${library}>")
    else()
        target_link_libraries(${target} PRIVATE
            "-Wl,--whole-archive"
            ${library}
            "-Wl,--no-whole-archive"
        )
    endif()
endfunction()

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
set(BERT_EXTRA_RUNTIME_DLL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/build/Release" CACHE PATH "Optional directory containing local runtime DLL/SO overrides")
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
require_path("${BERT_ONNXRUNTIME_SHARED_LIB}" "ONNX Runtime package")

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
if(NOT DEFINED LLAMA_CPP_ROOT)
    set(LLAMA_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp" CACHE PATH "Path to llama.cpp repository")
endif()
if(NOT DEFINED LLAMA_CPP_BUILD)
    set(LLAMA_CPP_BUILD "${LLAMA_CPP_ROOT}/build" CACHE PATH "Path to llama.cpp build directory")
endif()

set(LLAMA_CPP_INCLUDE_DIRS
    "${LLAMA_CPP_ROOT}/include"
    "${LLAMA_CPP_ROOT}/tools/mtmd"
    "${LLAMA_CPP_ROOT}/common"
    "${LLAMA_CPP_ROOT}/ggml/include"
)

if(WIN32)
    if(EXISTS "${LLAMA_CPP_BUILD}/src/Release/llama.lib")
        set(LLAMA_CPP_LIB_DIR "${LLAMA_CPP_BUILD}/src/Release")
        set(LLAMA_CPP_COMMON_LIB_DIR "${LLAMA_CPP_BUILD}/common/Release")
        set(LLAMA_CPP_MTMD_LIB_DIR "${LLAMA_CPP_BUILD}/tools/mtmd/Release")
        set(LLAMA_CPP_GGML_LIB_DIR "${LLAMA_CPP_BUILD}/ggml/src/Release")
        set(LLAMA_CPP_BIN_DIR "${LLAMA_CPP_BUILD}/bin/Release")
    else()
        set(LLAMA_CPP_LIB_DIR "${LLAMA_CPP_BUILD}/src")
        set(LLAMA_CPP_COMMON_LIB_DIR "${LLAMA_CPP_BUILD}/common")
        set(LLAMA_CPP_MTMD_LIB_DIR "${LLAMA_CPP_BUILD}/tools/mtmd")
        set(LLAMA_CPP_GGML_LIB_DIR "${LLAMA_CPP_BUILD}/ggml/src")
        set(LLAMA_CPP_BIN_DIR "${LLAMA_CPP_BUILD}/bin")
    endif()
    set(LLAMA_CPP_LIBS
        "${LLAMA_CPP_LIB_DIR}/llama.lib"
        "${LLAMA_CPP_MTMD_LIB_DIR}/mtmd.lib"
        "${LLAMA_CPP_COMMON_LIB_DIR}/llama-common.lib"
        "${LLAMA_CPP_GGML_LIB_DIR}/ggml.lib"
        "${LLAMA_CPP_GGML_LIB_DIR}/ggml-base.lib"
    )
    set(LLAMA_CPP_RUNTIME_FILES
        "${LLAMA_CPP_BIN_DIR}/llama.dll"
        "${LLAMA_CPP_BIN_DIR}/ggml.dll"
        "${LLAMA_CPP_BIN_DIR}/ggml-base.dll"
        "${LLAMA_CPP_BIN_DIR}/ggml-cpu.dll"
        "${LLAMA_CPP_BIN_DIR}/ggml-cuda.dll"
        "${LLAMA_CPP_BIN_DIR}/mtmd.dll"
        "${LLAMA_CPP_BIN_DIR}/llama-common.dll"
    )
elseif(UNIX AND NOT APPLE)
    set(LLAMA_CPP_LIB_DIR "${LLAMA_CPP_BUILD}/src")
    set(LLAMA_CPP_COMMON_LIB_DIR "${LLAMA_CPP_BUILD}/common")
    set(LLAMA_CPP_MTMD_LIB_DIR "${LLAMA_CPP_BUILD}/tools/mtmd")
    set(LLAMA_CPP_GGML_LIB_DIR "${LLAMA_CPP_BUILD}/ggml/src")
    set(LLAMA_CPP_BIN_DIR "${LLAMA_CPP_BUILD}/bin")
    set(LLAMA_CPP_LIBS
        "${LLAMA_CPP_LIB_DIR}/libllama.so"
        "${LLAMA_CPP_MTMD_LIB_DIR}/libmtmd.so"
        "${LLAMA_CPP_COMMON_LIB_DIR}/libllama-common.a"
        "${LLAMA_CPP_GGML_LIB_DIR}/libggml.so"
        "${LLAMA_CPP_GGML_LIB_DIR}/libggml-base.so"
    )
    file(GLOB LLAMA_CPP_RUNTIME_FILES
        "${LLAMA_CPP_BIN_DIR}/libllama.so*"
        "${LLAMA_CPP_BIN_DIR}/libggml*.so*"
        "${LLAMA_CPP_BIN_DIR}/libmtmd.so*"
        "${LLAMA_CPP_LIB_DIR}/libllama.so*"
        "${LLAMA_CPP_MTMD_LIB_DIR}/libmtmd.so*"
        "${LLAMA_CPP_GGML_LIB_DIR}/libggml*.so*"
    )
endif()

foreach(LLAMA_CPP_INCLUDE_DIR ${LLAMA_CPP_INCLUDE_DIRS})
    require_path("${LLAMA_CPP_INCLUDE_DIR}" "llama.cpp include directory")
endforeach()

foreach(LLAMA_CPP_LIB ${LLAMA_CPP_LIBS})
    require_path("${LLAMA_CPP_LIB}" "llama.cpp library")
endforeach()

if(WIN32)
    set(BERT_DEFAULT_OPENCV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/opencv-4.10.0-windows/build")
    set(OpenCV_DIR "${BERT_DEFAULT_OPENCV_ROOT}/x64/vc16/lib" CACHE PATH "Path to OpenCVConfig.cmake")
    set(BERT_OPENCV_RUNTIME_DIR "${BERT_DEFAULT_OPENCV_ROOT}/x64/vc16/bin")
    set(BERT_OPENCV_RUNTIME_GLOB "${BERT_OPENCV_RUNTIME_DIR}/opencv_world*.dll")
    set(BERT_OPENCV_EXTRA_RUNTIME_GLOB "${BERT_OPENCV_RUNTIME_DIR}/opencv_videoio_*.dll")
elseif(UNIX AND NOT APPLE)
    set(BERT_DEFAULT_OPENCV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/opencv-4.10.0-linux")
    set(OpenCV_DIR "${BERT_DEFAULT_OPENCV_ROOT}/lib/cmake/opencv4" CACHE PATH "Path to OpenCVConfig.cmake")
    set(BERT_OPENCV_RUNTIME_DIR "${BERT_DEFAULT_OPENCV_ROOT}/lib")
    set(BERT_OPENCV_RUNTIME_GLOB "${BERT_OPENCV_RUNTIME_DIR}/libopencv*.so*")
    set(BERT_OPENCV_EXTRA_RUNTIME_GLOB "")
endif()

find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc videoio)
file(GLOB BERT_OPENCV_RUNTIME_FILES
    "${BERT_OPENCV_RUNTIME_GLOB}"
    "${BERT_OPENCV_EXTRA_RUNTIME_GLOB}"
)

set(BERT_BOOST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/boost_1_85_0" CACHE PATH "Path to Boost header package")
require_path("${BERT_BOOST_ROOT}/boost/asio.hpp" "Boost.Asio header")
require_path("${BERT_BOOST_ROOT}/boost/beast.hpp" "Boost.Beast header")
require_path("${BERT_BOOST_ROOT}/boost/redis.hpp" "Boost.Redis header")
require_path("${BERT_BOOST_ROOT}/boost/redis/src.hpp" "Boost.Redis source header")

add_library(boost_asio_headers INTERFACE)
target_include_directories(boost_asio_headers INTERFACE "${BERT_BOOST_ROOT}")
if(WIN32)
    target_compile_definitions(boost_asio_headers INTERFACE _WIN32_WINNT=0x0602)
    target_link_libraries(boost_asio_headers INTERFACE ws2_32 mswsock)
endif()

add_library(boost_redis_headers INTERFACE)
target_link_libraries(boost_redis_headers INTERFACE boost_asio_headers OpenSSL::SSL OpenSSL::Crypto)

add_library(boost_interprocess_headers INTERFACE)
target_include_directories(boost_interprocess_headers INTERFACE "${BERT_BOOST_ROOT}")

set(BERT_EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen-5.0.1" CACHE PATH "Path to Eigen header package")
require_path("${BERT_EIGEN_ROOT}/Eigen/Dense" "Eigen headers")

add_library(eigen_headers INTERFACE)
target_include_directories(eigen_headers INTERFACE "${BERT_EIGEN_ROOT}")

set(BERT_SQLITE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/sqlite-amalgamation-3530100" CACHE PATH "Path to SQLite headers")
set(BERT_SQLITE_DLL "${CMAKE_CURRENT_SOURCE_DIR}/deps/sqlite3.dll" CACHE FILEPATH "Path to sqlite3.dll")
set(BERT_SQLITE_DEF "${CMAKE_CURRENT_SOURCE_DIR}/deps/sqlite3.def" CACHE FILEPATH "Path to sqlite3.def")
set(BERT_SQLITE_BUILD_DEF "${CMAKE_CURRENT_BINARY_DIR}/deps/sqlite3.def")
set(BERT_SQLITE_IMPLIB "${CMAKE_CURRENT_BINARY_DIR}/deps/sqlite3.lib")

require_path("${BERT_SQLITE_INCLUDE_DIR}/sqlite3.h" "SQLite header")
require_path("${BERT_SQLITE_DLL}" "SQLite runtime")
require_path("${BERT_SQLITE_DEF}" "SQLite module definition")

if(WIN32)
    add_custom_command(
        OUTPUT "${BERT_SQLITE_IMPLIB}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/deps"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${BERT_SQLITE_DEF}" "${BERT_SQLITE_BUILD_DEF}"
        COMMAND lib /def:sqlite3.def /machine:x64 /out:sqlite3.lib
        DEPENDS "${BERT_SQLITE_DEF}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/deps"
        VERBATIM
    )
    add_custom_target(sqlite_import_lib DEPENDS "${BERT_SQLITE_IMPLIB}")

    add_library(sqlite3 SHARED IMPORTED)
    add_dependencies(sqlite3 sqlite_import_lib)
    set_target_properties(sqlite3 PROPERTIES
        IMPORTED_LOCATION "${BERT_SQLITE_DLL}"
        IMPORTED_IMPLIB "${BERT_SQLITE_IMPLIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${BERT_SQLITE_INCLUDE_DIR}"
    )
else()
    message(FATAL_ERROR "SQLite prebuilt dependency currently expects the Windows package")
endif()

set(BERT_FAISS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/faiss-1.14.1-cpu-win64" CACHE PATH "Path to Faiss CPU package")
set(BERT_MKL_RUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/mkl-2023.1.0-win64" CACHE PATH "Path to MKL runtime package")

require_path("${BERT_FAISS_ROOT}/include/faiss/IndexFlat.h" "Faiss headers")
require_path("${BERT_FAISS_ROOT}/lib/faiss.lib" "Faiss import library")
require_path("${BERT_FAISS_ROOT}/bin/faiss.dll" "Faiss runtime")
require_path("${BERT_MKL_RUNTIME_ROOT}/Library/bin/mkl_rt.2.dll" "MKL runtime")

set(faiss_DIR "${BERT_FAISS_ROOT}/share/faiss" CACHE PATH "Path to Faiss config package")
find_package(faiss CONFIG REQUIRED)
set(BERT_FAISS_RUNTIME_FILES
    "${BERT_FAISS_ROOT}/bin/faiss.dll"
    "${BERT_MKL_RUNTIME_ROOT}/Library/bin/mkl_rt.2.dll"
)

# GStreamer prebuilt package for media/WebRTC probes and future media pipeline work.
set(GSTREAMER_ROOT "D:/Program Files/gstreamer/1.0/msvc_x86_64" CACHE PATH "Path to GStreamer MSVC x64 package")
if(EXISTS "${GSTREAMER_ROOT}/bin/gst-inspect-1.0.exe")
    add_library(gstreamer_headers INTERFACE)
    target_include_directories(gstreamer_headers INTERFACE
        "${GSTREAMER_ROOT}/include/gstreamer-1.0"
        "${GSTREAMER_ROOT}/include/glib-2.0"
        "${GSTREAMER_ROOT}/lib/glib-2.0/include"
    )

    add_library(gstreamer_core INTERFACE)
    target_link_libraries(gstreamer_core INTERFACE
        gstreamer_headers
        "${GSTREAMER_ROOT}/lib/gstreamer-1.0.lib"
        "${GSTREAMER_ROOT}/lib/gobject-2.0.lib"
        "${GSTREAMER_ROOT}/lib/glib-2.0.lib"
    )
else()
    message(STATUS "GStreamer package not found at ${GSTREAMER_ROOT}; media_gstreamer_probe will not be built")
endif()

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

add_library(agent_core STATIC
    src/core/blocking_queue.h
    src/core/exception.h
    src/core/logger_adapter.h
    src/core/memory_pool.cpp
    src/core/memory_pool.h
    src/core/object_pool.h
    src/core/result.h
    src/core/shared_memory_block.h
    src/core/thread_pool.cpp
    src/core/thread_pool.h
    src/core/unique_handle.h
)

target_include_directories(agent_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_link_libraries(agent_core PUBLIC spdlog::spdlog)

add_library(agent_net STATIC
    src/net/backpressure_queue.h
    src/net/connection_pool.cpp
    src/net/connection_pool.h
    src/net/http_request_filter.cpp
    src/net/http_request_filter.h
    src/net/http_server.cpp
    src/net/http_server.h
    src/net/http_types.h
    src/net/protocol_types.h
    src/net/request_interfaces.h
    src/net/shared_buffer.h
    src/net/static_file_handler.cpp
    src/net/static_file_handler.h
    src/net/websocket_session.cpp
    src/net/websocket_session.h
    src/net/websocket_types.h
)

target_include_directories(agent_net PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net
)

target_link_libraries(agent_net PUBLIC
    agent_core
)

if(TARGET boost_asio_headers)
    target_link_libraries(agent_net PUBLIC boost_asio_headers)
endif()

# ──────────── TLS context (reusable for HTTP, WebSocket, SMTP, etc.) ────────────
add_library(agent_tls STATIC
    src/net/tls/tls_options.h
    src/net/tls/tls_context.h
    src/net/tls/tls_context.cpp
)

target_include_directories(agent_tls PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net/tls
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_link_libraries(agent_tls PUBLIC
    agent_core
    boost_asio_headers
    OpenSSL::SSL
    OpenSSL::Crypto
)

if(BERT_BUILD_TESTS)
    include(GoogleTest)
    add_executable(tls_tests
        tests/net/tls/tls_context_test.cpp
    )
    target_link_libraries(tls_tests PRIVATE
        agent_tls
        GTest::gtest_main
    )
    gtest_discover_tests(tls_tests DISCOVERY_MODE PRE_TEST)
endif()

# ──────────── Outbound HTTP/HTTPS client (reusable for LLM, RAG, webhooks) ────────────
add_library(agent_http_client STATIC
    src/net/http_client/http_client.h
    src/net/http_client/url_parser.h
    src/net/http_client/url_parser.cpp
    src/net/http_client/retry_policy.h
    src/net/http_client/retry_policy.cpp
    src/net/http_client/beast_http_client.h
    src/net/http_client/beast_http_client.cpp
    src/net/http_client/concurrent_http_client.h
)

target_include_directories(agent_http_client PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net/http_client
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net/tls
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_link_libraries(agent_http_client PUBLIC
    agent_core
    agent_tls
    boost_asio_headers
    OpenSSL::SSL
    OpenSSL::Crypto
)

if(BERT_BUILD_TESTS)
    add_executable(http_client_tests
        tests/net/http_client/url_parser_test.cpp
        tests/net/http_client/retry_policy_test.cpp
        tests/net/http_client/beast_http_client_test.cpp
    )
    target_link_libraries(http_client_tests PRIVATE
        agent_http_client
        GTest::gtest_main
    )
    gtest_discover_tests(http_client_tests DISCOVERY_MODE PRE_TEST)
endif()

# ──────────── LLM client (OpenAI-compatible) ────────────
add_library(agent_llm STATIC
    src/llm/openai_llm_client.h
    src/llm/openai_llm_client.cpp
    src/llm/local_llm_client.h
    src/llm/local_llm_client.cpp
)

target_include_directories(agent_llm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net/http_client
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_link_libraries(agent_llm PUBLIC
    agent_http_client
    multimodal_proto
    gRPC::grpc++
)

if(BERT_BUILD_TESTS)
    add_executable(llm_tests
        tests/llm/openai_llm_client_test.cpp
    )
    target_link_libraries(llm_tests PRIVATE
        agent_llm
        GTest::gtest_main
    )
    gtest_discover_tests(llm_tests DISCOVERY_MODE PRE_TEST)

    add_executable(llm_integration_tests
        tests/llm/llm_integration_e2e_test.cpp
    )
    target_link_libraries(llm_integration_tests PRIVATE
        agent_llm
        agent_config
        GTest::gtest_main
    )
    link_whole_archive(llm_integration_tests agent_config)
    gtest_discover_tests(llm_integration_tests DISCOVERY_MODE PRE_TEST)
endif()

# ──────────── Manual smoke test against a real LLM API (not part of CTest) ────────────
add_executable(llm_smoke_test
    tools/llm_smoke_test.cpp
)
target_link_libraries(llm_smoke_test PRIVATE
    agent_llm
    agent_config
    agent_tls
)
link_whole_archive(llm_smoke_test agent_config)

# ──────────── L3 compression E2E test (not part of CTest) ────────────
add_executable(l3_compression_e2e_test
    tools/l3_compression_e2e_test.cpp
)
target_include_directories(l3_compression_e2e_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/memory
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage/sqlite
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage/vector
    ${CMAKE_CURRENT_SOURCE_DIR}/src/semantic_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vector
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
)
target_link_libraries(l3_compression_e2e_test PRIVATE
    agent_memory
    agent_llm
    agent_semantic_cache
    agent_storage
    agent_vector
    agent_vector_storage
    agent_config
    agent_tls
    nlohmann_json::nlohmann_json
)
link_whole_archive(l3_compression_e2e_test agent_config)
copy_runtime_files(l3_compression_e2e_test "${BERT_SQLITE_DLL}")

# ──────────── Persona Gateway frontend E2E server (not part of CTest) ────────────
add_executable(document_analysis_e2e_test
    tools/document_analysis_e2e_test.cpp
)
target_include_directories(document_analysis_e2e_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/document
    ${CMAKE_CURRENT_SOURCE_DIR}/src/document/ooxml
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/gateway
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/persona
    ${CMAKE_CURRENT_SOURCE_DIR}/src/semantic_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vector
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
)
target_link_libraries(document_analysis_e2e_test PRIVATE
    agent_service
    agent_document
    agent_vector
    agent_llm
    agent_tls
    agent_config
    nlohmann_json::nlohmann_json
    OpenSSL::SSL
    OpenSSL::Crypto
)
link_whole_archive(document_analysis_e2e_test agent_config)
copy_runtime_files(document_analysis_e2e_test "${BERT_SQLITE_DLL}")
copy_runtime_files(document_analysis_e2e_test ${VCPKG_RUNTIME_DLLS})

add_executable(persona_gateway_e2e_server
    tools/persona_gateway_e2e_server.cpp
)
target_include_directories(persona_gateway_e2e_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage/sqlite
)
target_link_libraries(persona_gateway_e2e_server PRIVATE
    server_runtime
    agent_service
    agent_llm
    agent_tls
    agent_config
    nlohmann_json::nlohmann_json
    OpenSSL::SSL
    OpenSSL::Crypto
)
link_whole_archive(persona_gateway_e2e_server agent_config)
copy_runtime_files(persona_gateway_e2e_server "${BERT_SQLITE_DLL}")

add_executable(agent_gateway_server
    src/server/main/agent_gateway_server.cpp
)
target_include_directories(agent_gateway_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage/sqlite
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage/vector
)
if(WIN32)
    target_include_directories(agent_gateway_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tools
    )
    target_link_libraries(agent_gateway_server PRIVATE Dbghelp)
endif()
target_link_libraries(agent_gateway_server PRIVATE
    server_runtime
    agent_service
    agent_document
    agent_vector
    agent_vector_storage
    agent_llm
    agent_tls
    agent_config
    nlohmann_json::nlohmann_json
    OpenSSL::SSL
    OpenSSL::Crypto
)
link_whole_archive(agent_gateway_server agent_config)
copy_runtime_files(agent_gateway_server "${BERT_SQLITE_DLL}")
copy_runtime_files(agent_gateway_server ${VCPKG_RUNTIME_DLLS})

add_executable(persona_config_migration
    tools/persona_config_migration.cpp
)
target_link_libraries(persona_config_migration PRIVATE
    nlohmann_json::nlohmann_json
)

add_executable(emotion_fusion_map_calibrator
    tools/emotion_fusion_map_calibrator.cpp
)
target_link_libraries(emotion_fusion_map_calibrator PRIVATE
    agent_service
    agent_models
    agent_vector
    agent_llm
    agent_http_client
    nlohmann_json::nlohmann_json
)
copy_runtime_files(emotion_fusion_map_calibrator "${BERT_ONNXRUNTIME_SHARED_LIB}")
copy_runtime_files(emotion_fusion_map_calibrator ${VCPKG_RUNTIME_DLLS})

# ──────────── Bare boost::redis smoke test (not part of CTest) ────────────
add_executable(redis_bare_connection_test
    tools/redis_bare_connection_test.cpp
)
target_link_libraries(redis_bare_connection_test PRIVATE
    agent_semantic_cache
)

add_executable(redis_stress_test
    tools/redis_stress_test.cpp
)
target_link_libraries(redis_stress_test PRIVATE
    agent_semantic_cache
)

# ──────────── Semantic cache pipeline (skeleton — Phase 5) ────────────
add_library(agent_semantic_cache STATIC
    src/semantic_cache/semantic_cache_types.h
    src/semantic_cache/isemantic_cache.h
    src/semantic_cache/context_risk_detector.h
    src/semantic_cache/context_risk_detector.cpp
    src/semantic_cache/semantic_cache_policy.h
    src/semantic_cache/semantic_cache_policy.cpp
    src/semantic_cache/redis_connection_pool.h
    src/semantic_cache/redis_connection_pool.cpp
    src/semantic_cache/semantic_cache_pipeline.h
    src/semantic_cache/semantic_cache_pipeline.cpp
    src/semantic_cache/l0_memory_cache_adapter.h
    src/semantic_cache/l0_memory_cache_adapter.cpp
)

target_include_directories(agent_semantic_cache PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/semantic_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage
)

target_link_libraries(agent_semantic_cache PUBLIC
    agent_core
    agent_vector
    agent_vector_storage
    redis++::redis++
    hiredis::hiredis
)

# ──────────── Document analysis infrastructure ────────────
add_library(agent_document STATIC
    src/document/chunk_builder.cpp
    src/document/document_file_store.h
    src/document/document_file_store.cpp
    src/document/document_analysis_service.h
    src/document/document_metadata_repository.h
    src/document/document_metadata_repository.cpp
    src/document/document_types.h
    src/document/document_llm_chunk_cache.h
    src/document/document_llm_chunk_cache.cpp
    src/document/document_pipeline.cpp
    src/document/diagnosis_builder.cpp
    src/document/mindmap_builder.cpp
    src/document/ooxml/ooxml_utils.cpp
    src/document/ooxml/ooxml_extractor.h
    src/document/ooxml/ooxml_extractor.cpp
)

target_include_directories(agent_document PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/document
    ${CMAKE_CURRENT_SOURCE_DIR}/src/document/ooxml
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/semantic_cache
)

target_link_libraries(agent_document PUBLIC
    agent_core
    agent_storage
    agent_llm
    agent_semantic_cache
    libzip::zip
    pugixml::pugixml
)

if(BERT_BUILD_TESTS)
    add_executable(semantic_cache_tests
        tests/semantic_cache/semantic_cache_pipeline_test.cpp
        tests/semantic_cache/test_reload_batch_cycle.cpp
    )

    target_link_libraries(semantic_cache_tests PRIVATE
        agent_semantic_cache
        agent_core
        GTest::gtest_main
    )

    gtest_discover_tests(semantic_cache_tests DISCOVERY_MODE PRE_TEST)

    add_executable(document_tests
        tests/document/ooxml_extractor_test.cpp
    )

    target_link_libraries(document_tests PRIVATE
        agent_document
        agent_storage
        GTest::gtest_main
    )

    copy_runtime_files(document_tests "${BERT_SQLITE_DLL}")
    copy_runtime_files(document_tests ${VCPKG_RUNTIME_DLLS})

    gtest_discover_tests(document_tests DISCOVERY_MODE PRE_TEST)

    add_executable(l0_benchmark
        tests/semantic_cache/l0_benchmark.cpp
    )

    target_link_libraries(l0_benchmark PRIVATE
        agent_semantic_cache
        agent_vector
        agent_core
        nlohmann_json::nlohmann_json
    )

    add_executable(full_pipeline_benchmark
        tests/semantic_cache/full_pipeline_benchmark.cpp
    )

    target_link_libraries(full_pipeline_benchmark PRIVATE
        agent_semantic_cache
        agent_vector
        agent_core
    )

    add_executable(l0_long_context_benchmark
        tests/semantic_cache/l0_long_context_benchmark.cpp
    )

    target_link_libraries(l0_long_context_benchmark PRIVATE
        agent_semantic_cache
        agent_vector
        agent_core
        nlohmann_json::nlohmann_json
    )

    add_executable(memory_tests
        tests/memory/long_term_memory_compressor_test.cpp
    )

    target_link_libraries(memory_tests PRIVATE
        agent_memory
        agent_semantic_cache
        agent_vector_storage
        agent_storage
        agent_llm
        GTest::gtest_main
    )

    copy_runtime_files(memory_tests "${BERT_SQLITE_DLL}")

    gtest_discover_tests(memory_tests DISCOVERY_MODE PRE_TEST)
endif()

# ──────────── Long-term memory (L3 compression) ────────────
add_library(agent_memory STATIC
    src/memory/long_term_memory_compressor.h
    src/memory/long_term_memory_compressor.cpp
)

target_include_directories(agent_memory PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/memory
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/semantic_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage
)

target_link_libraries(agent_memory PUBLIC
    agent_core
    agent_llm
    agent_semantic_cache
    agent_vector
    agent_vector_storage
    agent_storage
    nlohmann_json::nlohmann_json
)

add_library(agent_storage STATIC
    src/storage/sqlite/sqlite_async_executor.cpp
    src/storage/sqlite/sqlite_async_executor.h
    src/storage/sqlite/sqlite_connection.cpp
    src/storage/sqlite/sqlite_connection.h
    src/storage/sqlite/sqlite_connection_pool.cpp
    src/storage/sqlite/sqlite_connection_pool.h
    src/storage/sqlite/sqlite_error.cpp
    src/storage/sqlite/sqlite_internal.h
    src/storage/sqlite/sqlite_statement.cpp
    src/storage/sqlite/sqlite_statement.h
    src/storage/sqlite/sqlite_transaction.cpp
    src/storage/sqlite/sqlite_transaction.h
)

target_include_directories(agent_storage PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage
)

target_link_libraries(agent_storage PUBLIC
    agent_core
    sqlite3
)

add_library(agent_vector_storage STATIC
    src/storage/vector/vector_metadata.h
    src/storage/vector/vector_repository.h
    src/storage/vector/sqlite_vector_repository.h
    src/storage/vector/sqlite_vector_repository.cpp
    src/storage/vector/vector_fingerprint.h
    src/storage/vector/vector_fingerprint.cpp
    src/storage/vector/vector_partition_registry.h
    src/storage/vector/vector_partition_registry.cpp
)

target_include_directories(agent_vector_storage PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage
)

target_link_libraries(agent_vector_storage PUBLIC
    agent_core
    agent_storage
    OpenSSL::SSL
    OpenSSL::Crypto
)

add_library(agent_ipc STATIC
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

add_library(agent_media_inference STATIC
    src/media/inference_frame_backlog.cpp
    src/media/inference_frame_backlog.h
    src/media/inference_frame_coordinator.cpp
    src/media/inference_frame_coordinator.h
    src/media/inference_frame_ipc_receiver.cpp
    src/media/inference_frame_ipc_receiver.h
    src/media/vision_inference_interfaces.h
)

target_include_directories(agent_media_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/media
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
)

target_link_libraries(agent_media_inference PUBLIC
    agent_core
    agent_ipc
    spdlog::spdlog
)

if(TARGET gstreamer_core)
    add_library(agent_media STATIC
        src/media/frame_encoding.cpp
        src/media/frame_encoding.h
        src/media/inference_frame_gateway_producer.cpp
        src/media/inference_frame_gateway_producer.h
        src/media/opencv_frame_sampler.cpp
        src/media/opencv_frame_sampler.h
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
        gstreamer_core
        "${GSTREAMER_ROOT}/lib/gstapp-1.0.lib"
        "${GSTREAMER_ROOT}/lib/gstvideo-1.0.lib"
        "${GSTREAMER_ROOT}/lib/gstsdp-1.0.lib"
        "${GSTREAMER_ROOT}/lib/gstwebrtc-1.0.lib"
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

add_executable(static_site_test_server
    tests/net/static_site_server.cpp
)

target_link_libraries(static_site_test_server PRIVATE
    agent_net
)

add_executable(static_site_concurrency_bench
    tests/net/static_site_concurrency_bench.cpp
)

target_link_libraries(static_site_concurrency_bench PRIVATE
    agent_net
)

add_executable(net_stability_bench
    tests/net/net_stability_bench.cpp
)

target_link_libraries(net_stability_bench PRIVATE
    agent_net
)

add_executable(ws_download_server
    tests/net/ws_download_server.cpp
)

target_link_libraries(ws_download_server PRIVATE
    agent_net
)

add_executable(ws_download_client
    tests/net/ws_download_client.cpp
)

target_link_libraries(ws_download_client PRIVATE
    agent_net
)

if(TARGET agent_media)
    add_executable(webrtc_signaling_smoke_server
        tools/webrtc_signaling_smoke_server.cpp
    )

    target_link_libraries(webrtc_signaling_smoke_server PRIVATE
        agent_service
        agent_media
        agent_net
    )

    copy_runtime_files(webrtc_signaling_smoke_server ${BERT_OPENCV_RUNTIME_FILES})

    if(WIN32)
        set_target_properties(webrtc_signaling_smoke_server PROPERTIES
            VS_DEBUGGER_ENVIRONMENT "PATH=${GSTREAMER_ROOT}/bin;%PATH%;GST_PLUGIN_PATH=${GSTREAMER_ROOT}/lib/gstreamer-1.0"
        )
    endif()
endif()

add_custom_command(TARGET static_site_test_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/net/static_site"
        "$<TARGET_FILE_DIR:static_site_test_server>/static_site"
)

if(BERT_BUILD_TESTS)
    set(BERT_GTEST_ROOT "" CACHE PATH "Path to a prebuilt GTest package")
    if(BERT_GTEST_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH "${BERT_GTEST_ROOT}")
    endif()

    find_package(GTest CONFIG REQUIRED)
    include(GoogleTest)

    add_executable(core_tests
        tests/core/core_infra_test.cpp
    )

    target_link_libraries(core_tests PRIVATE
        agent_core
        GTest::gtest_main
    )

    gtest_discover_tests(core_tests DISCOVERY_MODE PRE_TEST)

    add_executable(media_inference_tests
        tests/media/inference_frame_backlog_test.cpp
        tests/media/inference_frame_coordinator_test.cpp
        tests/media/inference_frame_ipc_test.cpp
    )

    target_link_libraries(media_inference_tests PRIVATE
        agent_media_inference
        GTest::gtest_main
    )

    gtest_discover_tests(media_inference_tests DISCOVERY_MODE PRE_TEST)

    add_executable(media_inference_bench
        tests/media/inference_frame_backlog_bench.cpp
    )

    target_link_libraries(media_inference_bench PRIVATE
        agent_media_inference
    )

    add_executable(media_inference_ipc_bench
        tests/media/inference_frame_ipc_bench.cpp
    )

    target_link_libraries(media_inference_ipc_bench PRIVATE
        agent_media_inference
    )

    add_executable(net_tests
        tests/net/net_protocol_test.cpp
    )

    target_link_libraries(net_tests PRIVATE
        agent_net
        GTest::gtest_main
    )

    gtest_discover_tests(net_tests DISCOVERY_MODE PRE_TEST)

    if(TARGET agent_media)
        add_executable(media_tests
            tests/media/frame_encoding_test.cpp
            tests/media/opencv_frame_sampler_test.cpp
            tests/media/webrtc_media_pipeline_test.cpp
            tests/media/webrtc_signaling_handler_test.cpp
            tests/media/webrtc_bin_test.cpp
        )

        target_link_libraries(media_tests PRIVATE
            agent_media
            GTest::gtest_main
        )

        copy_runtime_files(media_tests ${BERT_OPENCV_RUNTIME_FILES})

        if(WIN32)
            set_target_properties(media_tests PROPERTIES
                VS_DEBUGGER_ENVIRONMENT "PATH=${GSTREAMER_ROOT}/bin;%PATH%;GST_PLUGIN_PATH=${GSTREAMER_ROOT}/lib/gstreamer-1.0"
            )
        endif()

        gtest_discover_tests(media_tests DISCOVERY_MODE PRE_TEST)

        add_executable(media_frame_encoding_bench
            tests/media/frame_encoding_bench.cpp
        )

        target_link_libraries(media_frame_encoding_bench PRIVATE
            agent_media
        )

        add_executable(media_rtc_ipc_fake_vlm_stress
            tests/media/rtc_ipc_fake_vlm_stress.cpp
        )

        target_link_libraries(media_rtc_ipc_fake_vlm_stress PRIVATE
            agent_media
            agent_media_inference
        )

        copy_runtime_files(media_rtc_ipc_fake_vlm_stress ${BERT_OPENCV_RUNTIME_FILES})

        if(WIN32)
            set_target_properties(media_rtc_ipc_fake_vlm_stress PROPERTIES
                VS_DEBUGGER_ENVIRONMENT "PATH=${GSTREAMER_ROOT}/bin;%PATH%;GST_PLUGIN_PATH=${GSTREAMER_ROOT}/lib/gstreamer-1.0"
            )
        endif()

        add_executable(media_ipc_e2e_peer
            tests/media/inference_frame_ipc_e2e_peer.cpp
        )

        target_link_libraries(media_ipc_e2e_peer PRIVATE
            agent_media_inference
        )

        add_executable(media_ipc_process_e2e_tests
            tests/media/inference_frame_ipc_process_e2e_test.cpp
        )

        target_link_libraries(media_ipc_process_e2e_tests PRIVATE
            agent_media
            GTest::gtest_main
        )

        target_compile_definitions(media_ipc_process_e2e_tests PRIVATE
            BOOST_PROCESS_USE_STD_FS
            MEDIA_IPC_E2E_PEER_PATH="$<TARGET_FILE:media_ipc_e2e_peer>"
        )

        add_dependencies(media_ipc_process_e2e_tests media_ipc_e2e_peer)
        gtest_discover_tests(media_ipc_process_e2e_tests DISCOVERY_MODE PRE_TEST)
    endif()

    add_executable(storage_tests
        tests/storage/sqlite_storage_test.cpp
    )

    target_link_libraries(storage_tests PRIVATE
        agent_storage
        GTest::gtest_main
    )

    copy_runtime_files(storage_tests "${BERT_SQLITE_DLL}")

    gtest_discover_tests(storage_tests DISCOVERY_MODE PRE_TEST)

    add_executable(vector_storage_tests
        tests/storage/vector/vector_storage_test.cpp
    )

    target_link_libraries(vector_storage_tests PRIVATE
        agent_vector_storage
        GTest::gtest_main
    )

    copy_runtime_files(vector_storage_tests "${BERT_SQLITE_DLL}")

    gtest_discover_tests(vector_storage_tests DISCOVERY_MODE PRE_TEST)

endif()

add_library(server_runtime STATIC
    src/server/runtime/logger.cpp
    src/server/runtime/logger.h
    src/server/runtime/server_common.cpp
    src/server/runtime/server_common.h
)

target_include_directories(server_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/server/runtime
)

target_link_libraries(server_runtime PUBLIC spdlog::spdlog)

add_library(agent_bert_models STATIC
    src/models/onnx_model.cpp
    src/models/onnx_model.h
    src/models/onnx_session_utils.cpp
    src/models/onnx_session_utils.h
)

target_include_directories(agent_bert_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/models
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${ONNXRUNTIME_ROOT}/include
)

target_link_libraries(agent_bert_models PUBLIC
    agent_core
    onnxruntime
    spdlog::spdlog
)

add_library(agent_models STATIC
    src/models/llama_handles.h
    src/models/llama_runner.cpp
    src/models/llama_runner.h
)

target_include_directories(agent_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/models
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${LLAMA_CPP_INCLUDE_DIRS}
)

target_link_libraries(agent_models PUBLIC
    agent_bert_models
    ${LLAMA_CPP_LIBS}
)

add_library(agent_cache STATIC
    src/cache/vlm_cache.cpp
    src/cache/vlm_cache.h
)

target_include_directories(agent_cache PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cache
)

target_link_libraries(agent_cache PUBLIC
    agent_models
    spdlog::spdlog
)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/HfTokenizersCapi.cmake")

add_library(agent_vector STATIC
    src/vector/vector_cache.cpp
    src/vector/vector_cache.h
    src/vector/hf_tokenizer.cpp
    src/vector/hf_tokenizer.h
    src/vector/tokenizer_pool.cpp
    src/vector/tokenizer_pool.h
    src/vector/text_embedding_model.h
    src/vector/text_embedding_pooling.cpp
    src/vector/onnx_text_embedding_model.cpp
    src/vector/onnx_text_embedding_model.h
    src/vector/embedding_pipeline.cpp
    src/vector/embedding_pipeline.h
    src/vector/vector_index.h
    src/vector/exact_vector_index.cpp
    src/vector/exact_vector_index.h
    src/vector/faiss_vector_index.cpp
    src/vector/faiss_vector_index.h
    src/vector/vector_index_manager.cpp
    src/vector/vector_index_manager.h
)

target_include_directories(agent_vector PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vector
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/models
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage
    ${ONNXRUNTIME_ROOT}/include
)

target_link_libraries(agent_vector PUBLIC
    agent_core
    agent_bert_models
    agent_vector_storage
    hf_tokenizers_capi
    faiss
    spdlog::spdlog
)

if(BERT_BUILD_TESTS)
    add_executable(vector_tests
        tests/vector/hf_tokenizer_test.cpp
        tests/vector/tokenizer_pool_test.cpp
        tests/vector/text_embedding_test.cpp
        tests/vector/vector_index_test.cpp
        tests/vector/vector_index_manager_test.cpp
    )

    target_link_libraries(vector_tests PRIVATE
        agent_vector
        agent_storage
        GTest::gtest_main
    )

    copy_runtime_files(vector_tests ${BERT_FAISS_RUNTIME_FILES})
    copy_runtime_files(vector_tests "${BERT_SQLITE_DLL}")

    file(GLOB VECTOR_TESTS_ONNXRUNTIME_LIBS "${ONNXRUNTIME_ROOT}/${BERT_ONNXRUNTIME_RUNTIME_GLOB}")
    foreach(_ort_lib ${VECTOR_TESTS_ONNXRUNTIME_LIBS})
        copy_runtime_files(vector_tests "${_ort_lib}")
    endforeach()

    gtest_discover_tests(vector_tests DISCOVERY_MODE PRE_TEST)
endif()

add_library(agent_config STATIC
    src/config/config_section.cpp
    src/config/config_section.h
    src/config/option_parser.cpp
    src/config/option_parser.h
    src/config/request_options.h
    src/config/sections/agent_gateway_config_section.cpp
    src/config/sections/auth_config_section.cpp
    src/config/sections/config_path_section.cpp
    src/config/sections/embedding_config_section.cpp
    src/config/sections/grpc_config_section.cpp
    src/config/sections/gateway_auth_config_section.cpp
    src/config/sections/http_config_section.cpp
    src/config/sections/limits_config_section.cpp
    src/config/sections/llm_config_section.cpp
    src/config/sections/model_config_section.cpp
    src/config/sections/persona_gateway_config_section.cpp
    src/config/sections/skill_session_config_section.cpp
    src/config/sections/vlm_cache_config_section.cpp
    src/config/sections/vram_config_section.cpp
    src/config/server_config.cpp
    src/config/server_config.h
    src/config/server_options.h
)

target_include_directories(agent_config PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(agent_config PUBLIC
    agent_net
    server_runtime
)

if(BERT_BUILD_TESTS)
    add_executable(config_tests
        tests/config/config_parser_test.cpp
    )

    target_link_libraries(config_tests PRIVATE
        agent_config
        GTest::gtest_main
    )
    link_whole_archive(config_tests agent_config)

    gtest_discover_tests(config_tests DISCOVERY_MODE PRE_TEST)
endif()

add_library(agent_service_core INTERFACE)

target_include_directories(agent_service_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/document
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/semantic_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/persona
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/gateway
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net
)

target_link_libraries(agent_service_core INTERFACE
    agent_core
    agent_net
    nlohmann_json::nlohmann_json
)

add_library(agent_agent_runtime STATIC
    src/service/persona/persona_algorithm.cpp
    src/service/persona/persona_algorithm.h
    src/service/persona/session_manager.cpp
    src/service/persona/session_manager.h
    src/service/persona/persona_runtime.cpp
    src/service/persona/persona_runtime.h
    src/service/persona/skill_session_manager.cpp
    src/service/persona/skill_session_manager.h
    src/service/persona/tool_memory_provider.cpp
    src/service/persona/tool_memory_provider.h
    src/service/persona/emotion_fusion_analyzer.cpp
    src/service/persona/emotion_fusion_analyzer.h
    src/service/persona/grpc_emotion_analyzer.cpp
    src/service/persona/grpc_emotion_analyzer.h
    src/service/gateway/classroom_scheduler.cpp
    src/service/gateway/classroom_scheduler.h
)

target_include_directories(agent_agent_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/persona
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/gateway
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/semantic_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/src/memory
)

target_link_libraries(agent_agent_runtime PUBLIC
    multimodal_proto
    agent_core
    agent_llm
    agent_semantic_cache
    agent_memory
    nlohmann_json::nlohmann_json
    gRPC::grpc++
    spdlog::spdlog
)

add_library(agent_gateway_auth STATIC
    src/service/gateway/gateway_auth.cpp
    src/service/gateway/gateway_auth.h
)

target_include_directories(agent_gateway_auth PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/gateway
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net
    ${CMAKE_CURRENT_SOURCE_DIR}/src/semantic_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage
)

target_link_libraries(agent_gateway_auth PUBLIC
    agent_core
    agent_net
    agent_semantic_cache
    agent_storage
    nlohmann_json::nlohmann_json
    OpenSSL::SSL
    OpenSSL::Crypto
    spdlog::spdlog
)

add_library(agent_agent_gateway STATIC
    src/service/gateway/gateway_models.h
    src/service/gateway/persona_gateway_service.cpp
    src/service/gateway/persona_gateway_service.h
    src/service/gateway/report_evaluator.h
    src/service/gateway/persona_gateway_agent_routes.cpp
    src/service/gateway/persona_gateway_route_core.h
    src/service/gateway/persona_gateway_route_helpers.h
)

target_include_directories(agent_agent_gateway PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/gateway
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/persona
    ${CMAKE_CURRENT_SOURCE_DIR}/src/semantic_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage
)

target_link_libraries(agent_agent_gateway PUBLIC
    agent_agent_runtime
    agent_gateway_auth
    agent_service_core
    agent_storage
    agent_semantic_cache
    agent_core
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

add_library(agent_classroom_gateway STATIC
    src/service/gateway/persona_gateway_classroom_routes.cpp
    src/service/gateway/persona_gateway_route_core.h
    src/service/gateway/persona_gateway_route_helpers.h
)

target_include_directories(agent_classroom_gateway PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/gateway
)

target_link_libraries(agent_classroom_gateway PUBLIC
    agent_agent_gateway
    agent_service_core
)

add_library(agent_training_report_gateway STATIC
    src/service/gateway/persona_gateway_teaching_routes.cpp
    src/service/gateway/persona_gateway_route_core.h
    src/service/gateway/persona_gateway_route_helpers.h
)

target_include_directories(agent_training_report_gateway PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/gateway
)

target_link_libraries(agent_training_report_gateway PUBLIC
    agent_agent_gateway
    agent_service_core
)

add_library(agent_document_gateway STATIC
    src/service/gateway/persona_gateway_document_routes.cpp
    src/service/gateway/persona_gateway_route_core.h
    src/service/gateway/persona_gateway_route_helpers.h
)

target_include_directories(agent_document_gateway PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/document
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/gateway
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/semantic_cache
)

target_link_libraries(agent_document_gateway PUBLIC
    agent_agent_gateway
    agent_document
    agent_llm
    agent_semantic_cache
    agent_service_core
)

add_library(agent_gateway_server_lib STATIC
    src/service/gateway/persona_gateway_http_adapter.cpp
    src/service/gateway/persona_gateway_http_adapter.h
    src/service/gateway/persona_gateway_route_core.h
    src/service/gateway/persona_gateway_route_helpers.h
    src/service/gateway/runtime_maintenance_service.cpp
    src/service/gateway/runtime_maintenance_service.h
    src/service/gateway/persona_gateway_server.cpp
    src/service/gateway/persona_gateway_server.h
)

target_include_directories(agent_gateway_server_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/document
    ${CMAKE_CURRENT_SOURCE_DIR}/src/document/ooxml
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/persona
    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/gateway
    ${CMAKE_CURRENT_SOURCE_DIR}/src/net
    ${CMAKE_CURRENT_SOURCE_DIR}/src/llm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/semantic_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/src/memory
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage
)

target_link_libraries(agent_gateway_server_lib PUBLIC
    agent_service_core
    agent_agent_gateway
    agent_gateway_auth
    agent_config
    agent_llm
    agent_semantic_cache
    agent_memory
    agent_storage
    agent_document
    agent_core
    agent_net
    nlohmann_json::nlohmann_json
    OpenSSL::SSL
    OpenSSL::Crypto
    spdlog::spdlog
)

if(TARGET agent_media)
    target_link_libraries(agent_gateway_server_lib PRIVATE
        agent_media
    )
endif()

add_library(agent_service INTERFACE)

target_link_libraries(agent_service INTERFACE
    agent_service_core
    agent_agent_runtime
    agent_gateway_auth
    agent_agent_gateway
    agent_classroom_gateway
    agent_training_report_gateway
    agent_document_gateway
    agent_gateway_server_lib
)

foreach(_gateway_route_target IN ITEMS
        document_analysis_e2e_test
        persona_gateway_e2e_server
        agent_gateway_server
        webrtc_signaling_smoke_server)
    if(TARGET ${_gateway_route_target})
        link_whole_archive(${_gateway_route_target} agent_agent_gateway)
        link_whole_archive(${_gateway_route_target} agent_classroom_gateway)
        link_whole_archive(${_gateway_route_target} agent_training_report_gateway)
        link_whole_archive(${_gateway_route_target} agent_document_gateway)
    endif()
endforeach()

if(TARGET agent_media)
    add_library(agent_skill_media STATIC
        src/service/persona/skill_vision_event_sink.cpp
        src/service/persona/skill_vision_event_sink.h
    )

    target_include_directories(agent_skill_media PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/service/persona
    )

    target_link_libraries(agent_skill_media PUBLIC
        agent_agent_runtime
        agent_media
    )
endif()

if(BERT_BUILD_TESTS)
    add_executable(service_tests
        tests/service/persona_algorithm_test.cpp
        tests/service/session_manager_test.cpp
        tests/service/persona_runtime_test.cpp
        tests/service/emotion_fusion_analyzer_test.cpp
        tests/service/persona_gateway_service_test.cpp
    )

    target_link_libraries(service_tests PRIVATE
        agent_service
        GTest::gtest_main
    )

    if(TARGET agent_skill_media)
        target_link_libraries(service_tests PRIVATE
            agent_skill_media
        )
    endif()

    target_include_directories(service_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tools
    )

    copy_runtime_files(service_tests "${BERT_SQLITE_DLL}")
    copy_runtime_files(service_tests ${BERT_FAISS_RUNTIME_FILES})
    copy_runtime_files(service_tests ${VCPKG_RUNTIME_DLLS})
    link_whole_archive(service_tests agent_agent_gateway)
    link_whole_archive(service_tests agent_classroom_gateway)
    link_whole_archive(service_tests agent_training_report_gateway)
    link_whole_archive(service_tests agent_document_gateway)

    gtest_discover_tests(service_tests DISCOVERY_MODE PRE_TEST)
endif()

# ==================== 推理服务端 ====================

option(BERT_BUILD_MULTIMODAL_INFERENCE_SERVER "Build full multimodal inference server with VLM/llama.cpp support" ON)
option(BERT_BUILD_EMOTION_INFERENCE_SERVER "Build CPU-only BERT emotion inference server" ON)

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

target_link_libraries(agent_emotion_server PUBLIC
    multimodal_proto
    agent_core
    agent_bert_models
    server_runtime
    agent_config
    gRPC::grpc++
    spdlog::spdlog
)

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
    link_whole_archive(multimodal_inference_server agent_config)

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

# ==================== 平台适配 ====================

if(UNIX AND NOT APPLE)
    set(BERT_RPATH_TARGETS bert_inference_client bert_benchmark_client)
    if(TARGET multimodal_inference_server)
        list(APPEND BERT_RPATH_TARGETS multimodal_inference_server)
    endif()
    if(TARGET emotion_inference_server)
        list(APPEND BERT_RPATH_TARGETS emotion_inference_server)
    endif()
    if(TARGET static_site_test_server)
        list(APPEND BERT_RPATH_TARGETS static_site_test_server)
    endif()
    if(TARGET static_site_concurrency_bench)
        list(APPEND BERT_RPATH_TARGETS static_site_concurrency_bench)
    endif()
    if(TARGET core_tests)
        list(APPEND BERT_RPATH_TARGETS core_tests)
    endif()
    if(TARGET net_tests)
        list(APPEND BERT_RPATH_TARGETS net_tests)
    endif()
    foreach(target ${BERT_RPATH_TARGETS})
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
        )
    endforeach()
endif()

# 安装
set(BERT_INSTALL_TARGETS bert_inference_client bert_benchmark_client)
if(TARGET multimodal_inference_server)
    list(APPEND BERT_INSTALL_TARGETS multimodal_inference_server)
endif()
if(TARGET emotion_inference_server)
    list(APPEND BERT_INSTALL_TARGETS emotion_inference_server)
endif()
install(TARGETS ${BERT_INSTALL_TARGETS} RUNTIME DESTINATION bin)

# 复制 ONNX Runtime 运行时库
file(GLOB ONNXRUNTIME_RUNTIME_LIBS "${ONNXRUNTIME_ROOT}/${BERT_ONNXRUNTIME_RUNTIME_GLOB}")
foreach(ONNXRUNTIME_RUNTIME_LIB ${ONNXRUNTIME_RUNTIME_LIBS})
    if(TARGET multimodal_inference_server)
        copy_runtime_files(multimodal_inference_server "${ONNXRUNTIME_RUNTIME_LIB}")
    endif()
    if(TARGET emotion_inference_server)
        copy_runtime_files(emotion_inference_server "${ONNXRUNTIME_RUNTIME_LIB}")
    endif()
    if(TARGET emotion_fusion_map_calibrator)
        copy_runtime_files(emotion_fusion_map_calibrator "${ONNXRUNTIME_RUNTIME_LIB}")
    endif()
endforeach()

if(EXISTS "${BERT_EXTRA_RUNTIME_DLL_ROOT}")
    if(WIN32)
        file(GLOB BERT_EXTRA_RUNTIME_LIBS
            "${BERT_EXTRA_RUNTIME_DLL_ROOT}/onnxruntime*.dll"
        )
    elseif(UNIX AND NOT APPLE)
        file(GLOB BERT_EXTRA_RUNTIME_LIBS
            "${BERT_EXTRA_RUNTIME_DLL_ROOT}/libonnxruntime*.so*"
        )
    endif()
    foreach(BERT_EXTRA_RUNTIME_LIB ${BERT_EXTRA_RUNTIME_LIBS})
        if(TARGET emotion_fusion_map_calibrator)
            copy_runtime_files(emotion_fusion_map_calibrator "${BERT_EXTRA_RUNTIME_LIB}")
        endif()
    endforeach()
endif()

# 复制 vcpkg 动态库依赖
if(WIN32 AND BERT_VCPKG_TRIPLET AND NOT VCPKG_RUNTIME_DLLS)
    file(GLOB VCPKG_RUNTIME_DLLS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${BERT_VCPKG_TRIPLET}/bin/*.dll")
    set(BERT_RUNTIME_COPY_TARGETS bert_inference_client bert_benchmark_client)
    if(TARGET multimodal_inference_server)
        list(APPEND BERT_RUNTIME_COPY_TARGETS multimodal_inference_server)
    endif()
    if(TARGET emotion_inference_server)
        list(APPEND BERT_RUNTIME_COPY_TARGETS emotion_inference_server)
    endif()
    if(TARGET static_site_test_server)
        list(APPEND BERT_RUNTIME_COPY_TARGETS static_site_test_server)
    endif()
    if(TARGET static_site_concurrency_bench)
        list(APPEND BERT_RUNTIME_COPY_TARGETS static_site_concurrency_bench)
    endif()
    if(TARGET core_tests)
        list(APPEND BERT_RUNTIME_COPY_TARGETS core_tests)
    endif()
    if(TARGET net_tests)
        list(APPEND BERT_RUNTIME_COPY_TARGETS net_tests)
    endif()
    if(TARGET storage_tests)
        list(APPEND BERT_RUNTIME_COPY_TARGETS storage_tests)
    endif()
    if(TARGET vector_tests)
        list(APPEND BERT_RUNTIME_COPY_TARGETS vector_tests)
    endif()
    if(TARGET config_tests)
        list(APPEND BERT_RUNTIME_COPY_TARGETS config_tests)
    endif()
    foreach(target ${BERT_RUNTIME_COPY_TARGETS})
        copy_runtime_files(${target} ${VCPKG_RUNTIME_DLLS})
    endforeach()
endif()

if(UNIX AND NOT APPLE AND BERT_VCPKG_TRIPLET)
    file(GLOB VCPKG_RUNTIME_SOS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${BERT_VCPKG_TRIPLET}/lib/*.so*")
    set(BERT_RUNTIME_COPY_TARGETS bert_inference_client bert_benchmark_client)
    if(TARGET multimodal_inference_server)
        list(APPEND BERT_RUNTIME_COPY_TARGETS multimodal_inference_server)
    endif()
    if(TARGET emotion_inference_server)
        list(APPEND BERT_RUNTIME_COPY_TARGETS emotion_inference_server)
    endif()
    if(TARGET static_site_test_server)
        list(APPEND BERT_RUNTIME_COPY_TARGETS static_site_test_server)
    endif()
    if(TARGET static_site_concurrency_bench)
        list(APPEND BERT_RUNTIME_COPY_TARGETS static_site_concurrency_bench)
    endif()
    if(TARGET core_tests)
        list(APPEND BERT_RUNTIME_COPY_TARGETS core_tests)
    endif()
    if(TARGET net_tests)
        list(APPEND BERT_RUNTIME_COPY_TARGETS net_tests)
    endif()
    if(TARGET storage_tests)
        list(APPEND BERT_RUNTIME_COPY_TARGETS storage_tests)
    endif()
    if(TARGET vector_tests)
        list(APPEND BERT_RUNTIME_COPY_TARGETS vector_tests)
    endif()
    if(TARGET config_tests)
        list(APPEND BERT_RUNTIME_COPY_TARGETS config_tests)
    endif()
    foreach(target ${BERT_RUNTIME_COPY_TARGETS})
        copy_runtime_files(${target} ${VCPKG_RUNTIME_SOS})
    endforeach()
endif()

# 复制 CUDA / cuDNN 运行时库（如果存在）
if(EXISTS "${CUDA_RUNTIME_DLL_ROOT}")
    if(WIN32)
        file(GLOB CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_DLL_ROOT}/*.dll")
    elseif(UNIX AND NOT APPLE)
        file(GLOB CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_DLL_ROOT}/*.so*")
    endif()
    if(TARGET multimodal_inference_server)
        copy_runtime_files(multimodal_inference_server ${CUDA_RUNTIME_LIBS})
    endif()
    if(TARGET emotion_fusion_map_calibrator)
        copy_runtime_files(emotion_fusion_map_calibrator ${CUDA_RUNTIME_LIBS})
    endif()
endif()

# Stable build-tree aliases for downstream projects using add_subdirectory().
function(agentloom_add_alias public_name internal_target)
    if(TARGET ${internal_target} AND NOT TARGET AgentLoom::${public_name})
        add_library(AgentLoom::${public_name} ALIAS ${internal_target})
    endif()
endfunction()

agentloom_add_alias(core agent_core)
agentloom_add_alias(net agent_net)
agentloom_add_alias(tls agent_tls)
agentloom_add_alias(http_client agent_http_client)
agentloom_add_alias(config agent_config)
agentloom_add_alias(storage agent_storage)
agentloom_add_alias(vector_storage agent_vector_storage)
agentloom_add_alias(vector agent_vector)
agentloom_add_alias(semantic_cache agent_semantic_cache)
agentloom_add_alias(memory agent_memory)
agentloom_add_alias(document agent_document)
agentloom_add_alias(llm agent_llm)
agentloom_add_alias(models agent_models)
agentloom_add_alias(cache agent_cache)
agentloom_add_alias(ipc agent_ipc)
agentloom_add_alias(media_inference agent_media_inference)
agentloom_add_alias(media agent_media)
agentloom_add_alias(runtime agent_agent_runtime)
agentloom_add_alias(gateway agent_gateway_server_lib)
agentloom_add_alias(service agent_service)
