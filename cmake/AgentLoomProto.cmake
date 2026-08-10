set(PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_DIR}")

function(agentloom_generate_proto proto_name)
    set(proto_file "${PROTO_DIR}/${proto_name}.proto")
    set(proto_src "${GENERATED_DIR}/${proto_name}.pb.cc")
    set(proto_hdr "${GENERATED_DIR}/${proto_name}.pb.h")
    set(grpc_src "${GENERATED_DIR}/${proto_name}.grpc.pb.cc")
    set(grpc_hdr "${GENERATED_DIR}/${proto_name}.grpc.pb.h")
    add_custom_command(
        OUTPUT "${proto_src}" "${proto_hdr}" "${grpc_src}" "${grpc_hdr}"
        COMMAND protobuf::protoc
        ARGS --cpp_out="${GENERATED_DIR}"
             --grpc_out="${GENERATED_DIR}"
             --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
             -I "${PROTO_DIR}" "${proto_file}"
        DEPENDS "${proto_file}"
        COMMENT "Generating ${proto_name} protobuf/gRPC code")
    set(${proto_name}_PROTO_SRC "${proto_src}" PARENT_SCOPE)
    set(${proto_name}_PROTO_HDR "${proto_hdr}" PARENT_SCOPE)
    set(${proto_name}_GRPC_SRC "${grpc_src}" PARENT_SCOPE)
    set(${proto_name}_GRPC_HDR "${grpc_hdr}" PARENT_SCOPE)
endfunction()

if(AGENTLOOM_BUILD_LEGACY_BERT_PROTO)
    agentloom_generate_proto(bert_inference)
endif()
agentloom_generate_proto(multimodal_inference)

# 兼容现有客户端与安装逻辑使用的变量名。
set(BERT_PROTO_SRC "${bert_inference_PROTO_SRC}")
set(BERT_PROTO_HDR "${bert_inference_PROTO_HDR}")
set(BERT_GRPC_SRC "${bert_inference_GRPC_SRC}")
set(BERT_GRPC_HDR "${bert_inference_GRPC_HDR}")
set(MULTIMODAL_PROTO_SRC "${multimodal_inference_PROTO_SRC}")
set(MULTIMODAL_PROTO_HDR "${multimodal_inference_PROTO_HDR}")
set(MULTIMODAL_GRPC_SRC "${multimodal_inference_GRPC_SRC}")
set(MULTIMODAL_GRPC_HDR "${multimodal_inference_GRPC_HDR}")

set(_agentloom_proto_outputs
    ${multimodal_inference_PROTO_SRC} ${multimodal_inference_PROTO_HDR}
    ${multimodal_inference_GRPC_SRC} ${multimodal_inference_GRPC_HDR})
if(AGENTLOOM_BUILD_LEGACY_BERT_PROTO)
    list(APPEND _agentloom_proto_outputs
        ${bert_inference_PROTO_SRC} ${bert_inference_PROTO_HDR}
        ${bert_inference_GRPC_SRC} ${bert_inference_GRPC_HDR})
endif()
add_custom_target(generate_proto ALL DEPENDS ${_agentloom_proto_outputs})

if(AGENTLOOM_BUILD_LEGACY_BERT_PROTO)
    add_library(bert_proto STATIC ${bert_inference_PROTO_SRC} ${bert_inference_GRPC_SRC})
    add_dependencies(bert_proto generate_proto)
    target_include_directories(bert_proto PUBLIC "${GENERATED_DIR}")
    target_link_libraries(bert_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
endif()

add_library(multimodal_proto STATIC ${multimodal_inference_PROTO_SRC} ${multimodal_inference_GRPC_SRC})
add_dependencies(multimodal_proto generate_proto)
target_include_directories(multimodal_proto PUBLIC "${GENERATED_DIR}")
target_link_libraries(multimodal_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
