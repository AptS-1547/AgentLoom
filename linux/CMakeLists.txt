cmake_minimum_required(VERSION 3.20)
project(AgentLoomLinuxControl LANGUAGES NONE)

find_program(BASH_EXECUTABLE bash REQUIRED)

set(LINUX_ROOT "${CMAKE_CURRENT_LIST_DIR}")

add_custom_target(linux_bootstrap
    COMMAND "${BASH_EXECUTABLE}" "${LINUX_ROOT}/scripts/bootstrap_toolchain.sh"
    WORKING_DIRECTORY "${LINUX_ROOT}/.."
    USES_TERMINAL
)

add_custom_target(linux_prepare
    COMMAND "${BASH_EXECUTABLE}" "${LINUX_ROOT}/scripts/prepare_deps.sh"
    WORKING_DIRECTORY "${LINUX_ROOT}/.."
    USES_TERMINAL
)

add_custom_target(linux_configure
    COMMAND "${BASH_EXECUTABLE}" "${LINUX_ROOT}/scripts/configure.sh"
    WORKING_DIRECTORY "${LINUX_ROOT}/.."
    USES_TERMINAL
)

add_custom_target(linux_build
    COMMAND "${BASH_EXECUTABLE}" "${LINUX_ROOT}/scripts/build.sh"
    WORKING_DIRECTORY "${LINUX_ROOT}/.."
    USES_TERMINAL
)

add_custom_target(linux_check
    COMMAND "${BASH_EXECUTABLE}" "${LINUX_ROOT}/scripts/check_artifacts.sh"
    WORKING_DIRECTORY "${LINUX_ROOT}/.."
    USES_TERMINAL
)

add_custom_target(linux_test
    COMMAND "${BASH_EXECUTABLE}" "${LINUX_ROOT}/scripts/test.sh" --skip-package
    WORKING_DIRECTORY "${LINUX_ROOT}/.."
    USES_TERMINAL
)

add_custom_target(linux_verify_package
    COMMAND "${BASH_EXECUTABLE}" "${LINUX_ROOT}/scripts/verify_package.sh"
    WORKING_DIRECTORY "${LINUX_ROOT}/.."
    USES_TERMINAL
)

add_custom_target(linux_package
    COMMAND "${BASH_EXECUTABLE}" "${LINUX_ROOT}/scripts/package.sh"
    WORKING_DIRECTORY "${LINUX_ROOT}/.."
    USES_TERMINAL
)

add_custom_target(linux_all
    DEPENDS linux_check linux_test linux_package
)

add_dependencies(linux_configure linux_prepare)
add_dependencies(linux_build linux_configure)
add_dependencies(linux_check linux_build)
add_dependencies(linux_test linux_build)
add_dependencies(linux_verify_package linux_build)
add_dependencies(linux_package linux_verify_package)
