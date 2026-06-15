from pathlib import Path
import shutil
import sys

source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
if target.exists():
    shutil.rmtree(target)

ignore = shutil.ignore_patterns(".git", ".vs", "build", "deps", "vcpkg", "vcpkg_installed")
shutil.copytree(source, target, ignore=ignore)

cmake = target / "CMakeLists.txt"
text = cmake.read_text(encoding="utf-8")
text = text.replace(
    "project(MultimodalInferenceService VERSION 2.0.0 LANGUAGES CXX)",
    "project(MultimodalInferenceService VERSION 2.0.0 LANGUAGES C CXX)",
)
text = text.replace(
    "target_link_libraries(agent_semantic_cache PUBLIC\n"
    "    agent_core\n",
    "target_compile_options(agent_semantic_cache PUBLIC -mavx2 -mfma)\n\n"
    "target_link_libraries(agent_semantic_cache PUBLIC\n"
    "    agent_core\n",
)
text = text.replace(
    'require_path("${BERT_SQLITE_DLL}" "SQLite runtime")\n'
    'require_path("${BERT_SQLITE_DEF}" "SQLite module definition")\n\n'
    'if(WIN32)',
    'if(WIN32)\n'
    '    require_path("${BERT_SQLITE_DLL}" "SQLite runtime")\n'
    '    require_path("${BERT_SQLITE_DEF}" "SQLite module definition")',
)
text = text.replace(
    'else()\n'
    '    message(FATAL_ERROR "SQLite prebuilt dependency currently expects the Windows package")\n'
    'endif()\n\n'
    'set(BERT_FAISS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/faiss-1.14.1-cpu-win64" CACHE PATH "Path to Faiss CPU package")\n'
    'set(BERT_MKL_RUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/mkl-2023.1.0-win64" CACHE PATH "Path to MKL runtime package")\n\n'
    'require_path("${BERT_FAISS_ROOT}/include/faiss/IndexFlat.h" "Faiss headers")\n'
    'require_path("${BERT_FAISS_ROOT}/lib/faiss.lib" "Faiss import library")\n'
    'require_path("${BERT_FAISS_ROOT}/bin/faiss.dll" "Faiss runtime")\n'
    'require_path("${BERT_MKL_RUNTIME_ROOT}/Library/bin/mkl_rt.2.dll" "MKL runtime")\n\n'
    'set(faiss_DIR "${BERT_FAISS_ROOT}/share/faiss" CACHE PATH "Path to Faiss config package")\n'
    'find_package(faiss CONFIG REQUIRED)\n'
    'set(BERT_FAISS_RUNTIME_FILES\n'
    '    "${BERT_FAISS_ROOT}/bin/faiss.dll"\n'
    '    "${BERT_MKL_RUNTIME_ROOT}/Library/bin/mkl_rt.2.dll"\n'
    ')\n',
    'elseif(UNIX AND NOT APPLE)\n'
    '    require_path("${BERT_SQLITE_INCLUDE_DIR}/sqlite3.c" "SQLite amalgamation source")\n'
    '    add_library(sqlite3 STATIC "${BERT_SQLITE_INCLUDE_DIR}/sqlite3.c")\n'
    '    target_include_directories(sqlite3 PUBLIC "${BERT_SQLITE_INCLUDE_DIR}")\n'
    '    target_link_libraries(sqlite3 PUBLIC dl pthread)\n'
    'endif()\n\n'
    'set(BERT_FAISS_ROOT "${BERT_FAISS_ROOT}" CACHE PATH "Path to Faiss CPU package")\n'
    'set(BERT_MKL_RUNTIME_ROOT "${BERT_MKL_RUNTIME_ROOT}" CACHE PATH "Path to MKL runtime package")\n'
    'require_path("${BERT_FAISS_ROOT}/include/faiss/IndexFlat.h" "Faiss headers")\n'
    'find_library(BERT_OPENBLAS_LIBRARY NAMES openblas libopenblas.so.0 REQUIRED)\n'
    'add_library(faiss SHARED IMPORTED)\n'
    'set_target_properties(faiss PROPERTIES\n'
    '    IMPORTED_LOCATION "${BERT_FAISS_ROOT}/lib/libfaiss.so"\n'
    '    INTERFACE_INCLUDE_DIRECTORIES "${BERT_FAISS_ROOT}/include"\n'
    '    INTERFACE_LINK_LIBRARIES "${BERT_OPENBLAS_LIBRARY}"\n'
    ')\n'
    'set(BERT_FAISS_RUNTIME_FILES\n'
    '    "${BERT_FAISS_ROOT}/lib/libfaiss.so"\n'
    '    "${BERT_MKL_RUNTIME_ROOT}/lib/libmkl_rt.so"\n'
    ')\n',
)

if "--disable-inference" in sys.argv:
    text = text.replace(
        "foreach(LLAMA_CPP_INCLUDE_DIR ${LLAMA_CPP_INCLUDE_DIRS})\n"
        '    require_path("${LLAMA_CPP_INCLUDE_DIR}" "llama.cpp include directory")\n'
        "endforeach()\n\n"
        "foreach(LLAMA_CPP_LIB ${LLAMA_CPP_LIBS})\n"
        '    require_path("${LLAMA_CPP_LIB}" "llama.cpp library")\n'
        "endforeach()\n",
        "if(BERT_BUILD_MULTIMODAL_INFERENCE_SERVER)\n"
        "    foreach(LLAMA_CPP_INCLUDE_DIR ${LLAMA_CPP_INCLUDE_DIRS})\n"
        '        require_path("${LLAMA_CPP_INCLUDE_DIR}" "llama.cpp include directory")\n'
        "    endforeach()\n\n"
        "    foreach(LLAMA_CPP_LIB ${LLAMA_CPP_LIBS})\n"
        '        require_path("${LLAMA_CPP_LIB}" "llama.cpp library")\n'
        "    endforeach()\n"
        "endif()\n",
    )

text = text.replace(
    "    target_link_libraries(llm_integration_tests PRIVATE\n"
    "        agent_llm\n"
    "        agent_config\n"
    "        GTest::gtest_main\n"
    "    )\n",
    "    target_link_libraries(llm_integration_tests PRIVATE\n"
    "        agent_llm\n"
    "        GTest::gtest_main\n"
    "    )\n",
)

text = text.replace(
    "target_link_libraries(llm_smoke_test PRIVATE\n"
    "    agent_llm\n"
    "    agent_config\n"
    "    agent_tls\n"
    ")\n",
    "target_link_libraries(llm_smoke_test PRIVATE\n"
    "    agent_llm\n"
    "    agent_tls\n"
    ")\n",
)

text = text.replace(
    "target_include_directories(agent_emotion_server PUBLIC\n"
    "    ${CMAKE_CURRENT_SOURCE_DIR}/src/server/grpc\n"
    "    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/inference\n"
    ")\n",
    "target_include_directories(agent_emotion_server PUBLIC\n"
    "    ${CMAKE_CURRENT_SOURCE_DIR}/src/server/grpc\n"
    "    ${CMAKE_CURRENT_SOURCE_DIR}/src/service/inference\n"
    "    ${CMAKE_CURRENT_SOURCE_DIR}/src/config\n"
    "    ${CMAKE_CURRENT_SOURCE_DIR}/third_party\n"
    ")\n",
)

text = text.replace(
    "target_link_libraries(agent_emotion_server PUBLIC\n"
    "    multimodal_proto\n"
    "    agent_core\n"
    "    agent_bert_models\n"
    "    server_runtime\n"
    "    agent_config\n"
    "    gRPC::grpc++\n"
    "    spdlog::spdlog\n"
    ")\n",
    "target_link_libraries(agent_emotion_server PUBLIC\n"
    "    multimodal_proto\n"
    "    agent_core\n"
    "    agent_bert_models\n"
    "    server_runtime\n"
    "    agent_net\n"
    "    gRPC::grpc++\n"
    "    spdlog::spdlog\n"
    ")\n",
)

linux_whole_archive_targets = (
    "l3_compression_e2e_test",
    "document_analysis_e2e_test",
    "persona_gateway_e2e_server",
    "agent_gateway_server",
    "multimodal_inference_server",
)
for target_name in linux_whole_archive_targets:
    text = text.replace(f"link_whole_archive({target_name} agent_config)\n", "")

text = text.replace(
    "    target_link_libraries(config_tests PRIVATE\n"
    "        agent_config\n"
    "        GTest::gtest_main\n"
    "    )\n"
    "    link_whole_archive(config_tests agent_config)\n",
    "    target_link_libraries(config_tests PRIVATE\n"
    "        GTest::gtest_main\n"
    "    )\n"
    "    link_whole_archive(config_tests agent_config)\n",
)
cmake.write_text(text, encoding="utf-8", newline="\n")
