cmake_minimum_required(VERSION 3.20)
project(BERTInferenceService VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# 源文件为 UTF-8，中文 Windows 默认代码页 936 会导致解析错误
add_compile_options(/utf-8)

# vcpkg 集成 - 直接指向已安装目录，不触发 manifest 重新安装
# 如需重新安装依赖，手动运行: vcpkg install --triplet=x64-windows
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

# onnxruntime: 使用预编译二进制（优先 GPU 包，不可用时回退 CPU 包）
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

set(ONNXRUNTIME_CPU_ROOT
    "${BERT_DEFAULT_ONNXRUNTIME_CPU_ROOT}"
    CACHE PATH "Path to the CPU ONNX Runtime package")
set(ONNXRUNTIME_GPU_ROOT
    "${BERT_DEFAULT_ONNXRUNTIME_GPU_ROOT}"
    CACHE PATH "Path to the GPU ONNX Runtime package")
set(CUDA_RUNTIME_DLL_ROOT
    "${BERT_DEFAULT_CUDA_RUNTIME_ROOT}"
    CACHE PATH "Directory containing CUDA/cuDNN runtime libraries for the ONNX Runtime CUDA provider")
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

# 生成 proto 代码
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

set(PROTO_FILE ${PROTO_DIR}/bert_inference.proto)
set(PROTO_SRC ${GENERATED_DIR}/bert_inference.pb.cc)
set(PROTO_HDR ${GENERATED_DIR}/bert_inference.pb.h)
set(GRPC_SRC ${GENERATED_DIR}/bert_inference.grpc.pb.cc)
set(GRPC_HDR ${GENERATED_DIR}/bert_inference.grpc.pb.h)

add_custom_command(
    OUTPUT ${PROTO_SRC} ${PROTO_HDR} ${GRPC_SRC} ${GRPC_HDR}
    COMMAND protobuf::protoc
    ARGS --cpp_out=${GENERATED_DIR}
         --grpc_out=${GENERATED_DIR}
         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         -I ${PROTO_DIR}
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating protobuf/gRPC code"
)

add_custom_target(generate_proto ALL DEPENDS ${PROTO_SRC} ${PROTO_HDR} ${GRPC_SRC} ${GRPC_HDR})

# proto 库
add_library(bert_proto STATIC ${PROTO_SRC} ${GRPC_SRC})
add_dependencies(bert_proto generate_proto)
target_include_directories(bert_proto PUBLIC ${GENERATED_DIR})
target_link_libraries(bert_proto PUBLIC
    protobuf::libprotobuf
    gRPC::grpc++
)

# 服务端
add_executable(bert_inference_server
    src/bert_inference_server.cpp
    src/logger.cpp
    src/logger.h
    src/onnx_model.cpp
    src/onnx_model.h
)

target_include_directories(bert_inference_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GENERATED_DIR}
    ${ONNXRUNTIME_ROOT}/include
)

target_link_libraries(bert_inference_server PRIVATE
    bert_proto
    onnxruntime
    gRPC::grpc++
    spdlog::spdlog
)

# 测试客户端
add_executable(bert_inference_client
    src/client_test.cpp
)

target_include_directories(bert_inference_client PRIVATE ${GENERATED_DIR})
target_link_libraries(bert_inference_client PRIVATE bert_proto gRPC::grpc++)

# 并发压测客户端
add_executable(bert_benchmark_client
    src/benchmark_client.cpp
)

target_include_directories(bert_benchmark_client PRIVATE ${GENERATED_DIR})
target_link_libraries(bert_benchmark_client PRIVATE bert_proto gRPC::grpc++)
if(UNIX AND NOT APPLE)
    foreach(target bert_inference_server bert_inference_client bert_benchmark_client)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
        )
    endforeach()
endif()

# 安装
install(TARGETS bert_inference_server bert_inference_client bert_benchmark_client RUNTIME DESTINATION bin)

# 复制 ONNX Runtime 运行时 DLL 到输出目录
file(GLOB ONNXRUNTIME_RUNTIME_DLLS "${ONNXRUNTIME_ROOT}/lib/onnxruntime*.dll")
file(GLOB ONNXRUNTIME_RUNTIME_LIBS "${ONNXRUNTIME_ROOT}/${BERT_ONNXRUNTIME_RUNTIME_GLOB}")
foreach(ONNXRUNTIME_RUNTIME_LIB ${ONNXRUNTIME_RUNTIME_LIBS})
    add_custom_command(TARGET bert_inference_server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_RUNTIME_LIB}"
            $<TARGET_FILE_DIR:bert_inference_server>
    )
endforeach()

# 复制 vcpkg 动态库依赖（gRPC / Protobuf / OpenSSL 等）
if(WIN32 AND BERT_VCPKG_TRIPLET)
    file(GLOB VCPKG_RUNTIME_DLLS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${BERT_VCPKG_TRIPLET}/bin/*.dll")
    foreach(VCPKG_RUNTIME_DLL ${VCPKG_RUNTIME_DLLS})
        add_custom_command(TARGET bert_inference_server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_RUNTIME_DLL}"
                $<TARGET_FILE_DIR:bert_inference_server>
        )
        add_custom_command(TARGET bert_inference_client POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_RUNTIME_DLL}"
                $<TARGET_FILE_DIR:bert_inference_client>
        )
        add_custom_command(TARGET bert_benchmark_client POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_RUNTIME_DLL}"
                $<TARGET_FILE_DIR:bert_benchmark_client>
        )
    endforeach()
endif()

if(UNIX AND NOT APPLE AND BERT_VCPKG_TRIPLET)
    file(GLOB VCPKG_RUNTIME_SOS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${BERT_VCPKG_TRIPLET}/lib/*.so*")
    foreach(VCPKG_RUNTIME_SO ${VCPKG_RUNTIME_SOS})
        add_custom_command(TARGET bert_inference_server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_RUNTIME_SO}"
                $<TARGET_FILE_DIR:bert_inference_server>
        )
        add_custom_command(TARGET bert_inference_client POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_RUNTIME_SO}"
                $<TARGET_FILE_DIR:bert_inference_client>
        )
        add_custom_command(TARGET bert_benchmark_client POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${VCPKG_RUNTIME_SO}"
                $<TARGET_FILE_DIR:bert_benchmark_client>
        )
    endforeach()
endif()

# 复制 CUDA / cuDNN 运行时 DLL（如果存在）
if(WIN32 AND EXISTS "${CUDA_RUNTIME_DLL_ROOT}")
    file(GLOB CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_DLL_ROOT}/*.dll")
    foreach(CUDA_RUNTIME_LIB ${CUDA_RUNTIME_LIBS})
        add_custom_command(TARGET bert_inference_server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CUDA_RUNTIME_LIB}"
                $<TARGET_FILE_DIR:bert_inference_server>
        )
    endforeach()
elseif(UNIX AND NOT APPLE AND EXISTS "${CUDA_RUNTIME_DLL_ROOT}")
    file(GLOB CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_DLL_ROOT}/*.so*")
    foreach(CUDA_RUNTIME_LIB ${CUDA_RUNTIME_LIBS})
        add_custom_command(TARGET bert_inference_server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CUDA_RUNTIME_LIB}"
                $<TARGET_FILE_DIR:bert_inference_server>
        )
    endforeach()
endif()
