#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

export VCPKG_INSTALL_ROOT="${VCPKG_INSTALL_ROOT:-$REPO_ROOT/build/linux-vcpkg-installed}"
export VCPKG_TRIPLET="${VCPKG_TRIPLET:-x64-linux-release}"
export VCPKG_BINARY_CACHE="${VCPKG_BINARY_CACHE:-$REPO_ROOT/build/vcpkg-binary-cache}"
export PACKAGE_DIR="${PACKAGE_DIR:-$REPO_ROOT/build/linux-package}"

include_inference=false
for arg in "$@"; do
    case "$arg" in
        --inference)
            include_inference=true
            ;;
        *)
            echo "unknown option: $arg" >&2
            exit 64
            ;;
    esac
done

bash linux/scripts/bootstrap_toolchain.sh
bash linux/scripts/prepare_deps.sh

if [[ "$include_inference" == true ]]; then
    bash linux/scripts/configure.sh --inference
    bash linux/scripts/build.sh --inference
    bash linux/scripts/package.sh --inference
else
    bash linux/scripts/configure.sh
    bash linux/scripts/build.sh
    bash linux/scripts/package.sh
fi

echo "[docker-build] Linux package created at $PACKAGE_DIR"
