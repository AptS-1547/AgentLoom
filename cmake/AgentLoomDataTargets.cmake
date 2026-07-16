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

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
        CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_compile_options(agent_semantic_cache PUBLIC -mavx2 -mfma)
endif()

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
