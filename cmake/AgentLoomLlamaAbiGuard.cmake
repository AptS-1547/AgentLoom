function(agentloom_classify_msvc_abi configuration output_variable)
    string(TOUPPER "${configuration}" normalized_configuration)
    if(normalized_configuration STREQUAL "DEBUG")
        set(abi_class "Debug")
    elseif(normalized_configuration STREQUAL "RELEASE"
            OR normalized_configuration STREQUAL "RELWITHDEBINFO"
            OR normalized_configuration STREQUAL "MINSIZEREL")
        set(abi_class "Release")
    else()
        set(abi_class "Unknown")
    endif()
    set(${output_variable} "${abi_class}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED AGENT_CONFIG OR NOT AGENT_CONFIG)
    message(FATAL_ERROR
        "AgentLoom build configuration is empty; cannot validate llama.cpp ABI compatibility.")
endif()
if(NOT DEFINED LLAMA_CONFIG OR NOT LLAMA_CONFIG)
    message(FATAL_ERROR
        "llama.cpp build configuration is empty; set LLAMA_CPP_PREBUILT_CONFIG explicitly.")
endif()

agentloom_classify_msvc_abi("${AGENT_CONFIG}" agent_abi_class)
agentloom_classify_msvc_abi("${LLAMA_CONFIG}" llama_abi_class)

if(agent_abi_class STREQUAL "Unknown")
    message(FATAL_ERROR
        "Unsupported AgentLoom configuration '${AGENT_CONFIG}' for llama.cpp ABI validation.")
endif()
if(llama_abi_class STREQUAL "Unknown")
    message(FATAL_ERROR
        "Unsupported llama.cpp configuration '${LLAMA_CONFIG}' at '${LLAMA_BUILD}'.")
endif()

if(NOT agent_abi_class STREQUAL llama_abi_class)
    message(FATAL_ERROR
        "llama.cpp ABI mismatch: AgentLoom '${AGENT_CONFIG}' uses the ${agent_abi_class} MSVC CRT, "
        "but prebuilt llama.cpp '${LLAMA_CONFIG}' at '${LLAMA_BUILD}' uses the ${llama_abi_class} CRT. "
        "Rebuild llama.cpp with a matching configuration or select a matching LLAMA_CPP_BUILD.")
endif()

message(STATUS
    "llama.cpp ABI check passed: AgentLoom=${AGENT_CONFIG}, llama.cpp=${LLAMA_CONFIG}, "
    "class=${agent_abi_class}")
