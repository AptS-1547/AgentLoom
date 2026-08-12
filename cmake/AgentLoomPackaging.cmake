include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(AGENTLOOM_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/AgentLoom")
set(_agentloom_public_targets
    bert_proto multimodal_proto agent_core agent_net agent_tls agent_http_client
    agent_llm agent_semantic_cache agent_document agent_memory agent_storage
    agent_vector_storage agent_conversation agent_ipc agent_ipc_grpc agent_media_inference agent_media_vlm_grpc agent_media
    server_runtime agent_bert_models agent_models agent_cache agent_vector agent_config
    agent_service_core agent_agent_runtime agent_gateway_auth agent_agent_gateway
    agent_classroom_gateway agent_training_report_gateway agent_document_gateway
    agent_gateway_server_lib agent_service agent_skill_media agent_grpc_runtime agent_emotion_server agent_server
)

set(_agentloom_public_target_names
    bert_proto multimodal_proto core net tls http_client llm semantic_cache document
    memory storage vector_storage conversation ipc ipc_grpc media_inference vlm_client media runtime bert_models
    models cache vector config service_core agent_runtime gateway_auth agent_gateway
    classroom_gateway training_report_gateway document_gateway gateway service skill_media grpc_runtime
    emotion_server server
)

if(NOT AGENTLOOM_BUILD_LOCAL_LLM)
    list(REMOVE_ITEM _agentloom_public_targets agent_models agent_cache agent_server)
    list(REMOVE_ITEM _agentloom_public_target_names models cache server)
endif()

list(LENGTH _agentloom_public_targets _agentloom_target_count)
list(LENGTH _agentloom_public_target_names _agentloom_name_count)
if(NOT _agentloom_target_count EQUAL _agentloom_name_count)
    message(FATAL_ERROR "AgentLoom package target lists are out of sync")
endif()

math(EXPR _agentloom_last_index "${_agentloom_target_count} - 1")
set(AGENTLOOM_PUBLIC_TARGETS "")
foreach(_agentloom_index RANGE ${_agentloom_last_index})
    list(GET _agentloom_public_targets ${_agentloom_index} _agentloom_target)
    list(GET _agentloom_public_target_names ${_agentloom_index} _agentloom_name)
    if(TARGET ${_agentloom_target})
        set_property(TARGET ${_agentloom_target} PROPERTY EXPORT_NAME ${_agentloom_name})
        list(APPEND AGENTLOOM_PUBLIC_TARGETS ${_agentloom_target})
    endif()
endforeach()

set(_agentloom_support_targets
    boost_asio_headers
    boost_redis_headers
    boost_interprocess_headers
    eigen_headers)
if(AGENTLOOM_BUILD_LOCAL_LLM)
    list(APPEND _agentloom_support_targets agentloom_llama_cpp_dependency)
endif()
if(TARGET agentloom_gstreamer_dependency)
    list(APPEND _agentloom_support_targets agentloom_gstreamer_dependency)
endif()
foreach(_agentloom_target IN LISTS _agentloom_support_targets)
    if(TARGET ${_agentloom_target})
        set_property(TARGET ${_agentloom_target} PROPERTY EXPORT_NAME dependency_${_agentloom_target})
        list(APPEND AGENTLOOM_PUBLIC_TARGETS ${_agentloom_target})
    endif()
endforeach()

if(TARGET sqlite3)
    get_target_property(_agentloom_sqlite_imported sqlite3 IMPORTED)
    if(NOT _agentloom_sqlite_imported)
        set_property(TARGET sqlite3 PROPERTY EXPORT_NAME dependency_sqlite3)
        list(APPEND AGENTLOOM_PUBLIC_TARGETS sqlite3)
    endif()
endif()

set(_agentloom_install_include_dirs
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/generated"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/core"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/net"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/net/tls"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/net/http_client"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/llm"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/config"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/document"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/document/ooxml"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/conversation"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/memory"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/storage"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/storage/sqlite"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/storage/vector"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/vector"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/semantic_cache"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/ipc"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/media"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/models"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/cache"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/service/persona"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/service/gateway"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/service/inference"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/server/runtime"
    "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/server/grpc")

function(_agentloom_make_include_interface_relocatable target)
    get_target_property(_build_include_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
    set(_relocatable_include_dirs "")
    if(_build_include_dirs)
        foreach(_include_dir IN LISTS _build_include_dirs)
            if(_include_dir MATCHES "^\\$<")
                list(APPEND _relocatable_include_dirs "${_include_dir}")
            else()
                list(APPEND _relocatable_include_dirs "$<BUILD_INTERFACE:${_include_dir}>")
            endif()
        endforeach()
    endif()
    foreach(_include_dir IN LISTS _agentloom_install_include_dirs)
        list(APPEND _relocatable_include_dirs "$<INSTALL_INTERFACE:${_include_dir}>")
    endforeach()
    set_property(TARGET ${target} PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${_relocatable_include_dirs}")
endfunction()

foreach(_agentloom_target IN LISTS AGENTLOOM_PUBLIC_TARGETS)
    _agentloom_make_include_interface_relocatable(${_agentloom_target})
endforeach()

set(_agentloom_sdk_targets "")
foreach(_agentloom_target IN LISTS AGENTLOOM_PUBLIC_TARGETS)
    get_target_property(_agentloom_imported ${_agentloom_target} IMPORTED)
    get_target_property(_agentloom_type ${_agentloom_target} TYPE)
    if(NOT _agentloom_imported AND NOT _agentloom_type STREQUAL "INTERFACE_LIBRARY")
        list(APPEND _agentloom_sdk_targets ${_agentloom_target})
    endif()
endforeach()
add_custom_target(agentloom_sdk DEPENDS ${_agentloom_sdk_targets})

foreach(_agentloom_target IN LISTS AGENTLOOM_PUBLIC_TARGETS)
    if(TARGET ${_agentloom_target})
        get_target_property(_agentloom_type ${_agentloom_target} TYPE)
        if(_agentloom_type STREQUAL "INTERFACE_LIBRARY")
            install(TARGETS ${_agentloom_target}
                EXPORT AgentLoomTargets
                INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
        else()
            install(TARGETS ${_agentloom_target}
                EXPORT AgentLoomTargets
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
        endif()
    endif()
endforeach()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

if(EXISTS "${GENERATED_DIR}")
    install(DIRECTORY "${GENERATED_DIR}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/AgentLoom/generated"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
endif()

install(EXPORT AgentLoomTargets
    FILE AgentLoomTargets.cmake
    NAMESPACE AgentLoom::
    DESTINATION "${AGENTLOOM_CMAKE_INSTALL_DIR}")

export(EXPORT AgentLoomTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/AgentLoomTargets.cmake"
    NAMESPACE AgentLoom::)

if(TARGET agent_media)
    set(AGENTLOOM_PACKAGE_HAS_MEDIA ON)
else()
    set(AGENTLOOM_PACKAGE_HAS_MEDIA OFF)
endif()

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/AgentLoomConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/AgentLoomConfig.cmake"
    INSTALL_DESTINATION "${AGENTLOOM_CMAKE_INSTALL_DIR}")

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/AgentLoomConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/AgentLoomConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/AgentLoomConfigVersion.cmake"
    DESTINATION "${AGENTLOOM_CMAKE_INSTALL_DIR}")
