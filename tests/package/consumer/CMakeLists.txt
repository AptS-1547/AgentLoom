cmake_minimum_required(VERSION 3.20)
project(AgentLoomPackageConsumer LANGUAGES CXX)

find_package(AgentLoom CONFIG REQUIRED)

add_executable(agentloom_package_consumer main.cpp)
target_compile_features(agentloom_package_consumer PRIVATE cxx_std_20)
target_link_libraries(agentloom_package_consumer PRIVATE
    AgentLoom::core
    AgentLoom::runtime
    AgentLoom::gateway)
