#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

require_linux
require_command ctest
[[ -f "$BUILD_DIR/CTestTestfile.cmake" ]] || fail "tests are not configured; run linux/scripts/configure.sh first"

run_all=false
include_redis=false

for arg in "$@"; do
    case "$arg" in
        --all)
            run_all=true
            ;;
        --redis)
            include_redis=true
            ;;
        *)
            fail "unknown test option: $arg"
            ;;
    esac
done

ctest_args=(--test-dir "$BUILD_DIR" -C Release --output-on-failure)

if [[ "$run_all" != true ]]; then
    excludes=("NOT_BUILT")
    if [[ "$include_redis" != true ]]; then
        excludes+=("ReloadBatchCycle")
    fi

    exclude_regex="$(IFS='|'; echo "${excludes[*]}")"
    ctest_args+=(-E "$exclude_regex")
fi

ctest "${ctest_args[@]}"
