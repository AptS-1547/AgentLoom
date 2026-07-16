option(BERT_BUILD_TESTS "Build C++ unit tests" OFF)
option(BERT_BUILD_MULTIMODAL_INFERENCE_SERVER "Build full multimodal inference server with VLM/llama.cpp support" ON)
option(BERT_BUILD_EMOTION_INFERENCE_SERVER "Build CPU-only BERT emotion inference server" ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

if(BERT_BUILD_TESTS)
    enable_testing()
endif()

if(WIN32)
    set(BERT_VCPKG_TRIPLET_DEFAULT "x64-windows")
elseif(UNIX AND NOT APPLE)
    set(BERT_VCPKG_TRIPLET_DEFAULT "x64-linux")
else()
    set(BERT_VCPKG_TRIPLET_DEFAULT "")
endif()

set(BERT_VCPKG_TRIPLET "${BERT_VCPKG_TRIPLET_DEFAULT}" CACHE STRING "vcpkg target triplet for dependency lookup")
if(BERT_VCPKG_TRIPLET)
    list(APPEND CMAKE_PREFIX_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${BERT_VCPKG_TRIPLET}"
        "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${BERT_VCPKG_TRIPLET}/share")
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
                    $<TARGET_FILE_DIR:${target}>)
        endif()
    endforeach()
endfunction()

function(link_whole_archive target library)
    if(MSVC)
        target_link_options(${target} PRIVATE "/WHOLEARCHIVE:$<TARGET_FILE:${library}>")
    else()
        target_link_libraries(${target} PRIVATE
            "-Wl,--whole-archive" ${library} "-Wl,--no-whole-archive")
    endif()
endfunction()
