#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINUX_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$LINUX_DIR/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build/linux-local-tests}"
LINUX_SOURCE_DIR="${LINUX_SOURCE_DIR:-$REPO_ROOT/build/linux-source}"
LINUX_INSTALL_DIR="${LINUX_INSTALL_DIR:-$REPO_ROOT/build/linux-agentloom-install}"
PACKAGE_CONSUMER_BUILD_DIR="${PACKAGE_CONSUMER_BUILD_DIR:-$REPO_ROOT/build/linux-package-consumer}"
LINUX_VENV_DIR="${LINUX_VENV_DIR:-$REPO_ROOT/build/linux-python-venv}"
PYTHON_BIN="${PYTHON_BIN:-$LINUX_VENV_DIR/bin/python}"
DEPS_DIR="${DEPS_DIR:-$REPO_ROOT/deps}"
VCPKG_ROOT="${VCPKG_ROOT:-$REPO_ROOT/vcpkg}"
VCPKG_INSTALL_ROOT="${VCPKG_INSTALL_ROOT:-$REPO_ROOT/build/linux-vcpkg-installed}"
VCPKG_BINARY_CACHE="${VCPKG_BINARY_CACHE:-$REPO_ROOT/build/vcpkg-binary-cache}"
VCPKG_TRIPLET="${VCPKG_TRIPLET:-x64-linux-release}"
BUILD_JOBS="${BUILD_JOBS:-2}"

if [[ -d "$HOME/.cargo/bin" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
fi

ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.17.1}"
ONNXRUNTIME_ROOT="$DEPS_DIR/onnxruntime-linux-x64-$ONNXRUNTIME_VERSION"
ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v$ONNXRUNTIME_VERSION/onnxruntime-linux-x64-$ONNXRUNTIME_VERSION.tgz"

SQLITE_VERSION="${SQLITE_VERSION:-3530100}"
SQLITE_ROOT="$DEPS_DIR/sqlite-amalgamation-$SQLITE_VERSION"
SQLITE_URL="https://www.sqlite.org/2025/sqlite-amalgamation-$SQLITE_VERSION.zip"

BOOST_VERSION="${BOOST_VERSION:-1_85_0}"
BOOST_ROOT="$DEPS_DIR/boost_$BOOST_VERSION"
BOOST_URL="https://archives.boost.io/release/1.85.0/source/boost_$BOOST_VERSION.tar.gz"

EIGEN_VERSION="${EIGEN_VERSION:-5.0.1}"
EIGEN_ROOT="$DEPS_DIR/eigen-$EIGEN_VERSION"
EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.zip"

FAISS_ROOT="$DEPS_DIR/faiss-1.14.1-cpu-linux"
FAISS_URL="${FAISS_URL:-https://conda.anaconda.org/conda-forge/linux-64/libfaiss-1.10.0-cpu_openblas_hfcc2109_0.conda}"
MKL_ROOT="$DEPS_DIR/mkl-2023.1.0-linux"
MKL_URL="${MKL_URL:-https://conda.anaconda.org/conda-forge/linux-64/mkl-2023.1.0-h84fe81f_48680.conda}"

LLAMA_CPP_TAG="${LLAMA_CPP_TAG:-b9360}"
LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$DEPS_DIR/llama.cpp}"
LLAMA_CPP_BUILD="${LLAMA_CPP_BUILD:-$LLAMA_CPP_ROOT/build}"
LLAMA_CPP_CUDA_URL="${LLAMA_CPP_CUDA_URL:-}"

log() {
    printf '[linux-build] %s\n' "$*"
}

fail() {
    printf '[linux-build] error: %s\n' "$*" >&2
    exit 1
}

require_linux() {
    [[ "$(uname -s)" == "Linux" ]] || fail "Linux or WSL2 is required"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

download_file() {
    local url="$1"
    local path="$2"
    if [[ -s "$path" ]]; then
        log "using cached archive $path"
        return 0
    fi
    rm -f "$path" "$path.tmp"
    log "downloading $url"
    curl --fail --location --retry 5 --retry-delay 5 --connect-timeout 30 "$url" -o "$path.tmp"
    test -s "$path.tmp"
    mv "$path.tmp" "$path"
}

ensure_zip() {
    local url="$1"
    local path="$2"
    if [[ -s "$path" ]] && ! unzip -tq "$path" >/dev/null; then
        rm -f "$path"
    fi
    download_file "$url" "$path"
    unzip -tq "$path" >/dev/null
}

ensure_tgz() {
    local url="$1"
    local path="$2"
    if [[ -s "$path" ]] && ! tar -tzf "$path" >/dev/null; then
        rm -f "$path"
    fi
    download_file "$url" "$path"
    tar -tzf "$path" >/dev/null
}

ensure_conda() {
    local url="$1"
    local path="$2"
    if [[ -s "$path" ]] && ! "$PYTHON_BIN" - "$path" <<'PY'
import sys
import zipfile
raise SystemExit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)
PY
    then
        rm -f "$path"
    fi
    download_file "$url" "$path"
}

extract_conda() {
    local archive="$1"
    local output="$2"
    "$PYTHON_BIN" - "$archive" "$output" <<'PY'
import pathlib
import shutil
import sys
import tarfile
import zipfile
import zstandard

archive = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
work = archive.with_suffix(".extract")
if work.exists():
    shutil.rmtree(work)
if output.exists():
    shutil.rmtree(output)
work.mkdir(parents=True)
output.mkdir(parents=True)
with zipfile.ZipFile(archive) as package:
        package.extractall(work)
payload = next(work.glob("pkg-*.tar.zst"))
tar_path = work / "payload.tar"
with payload.open("rb") as compressed, tar_path.open("wb") as decompressed:
    reader = zstandard.ZstdDecompressor().stream_reader(compressed)
    shutil.copyfileobj(reader, decompressed)
with tarfile.open(tar_path) as package:
    package.extractall(output)
shutil.rmtree(work)
PY
}

has_cuda_llama_binary() {
    [[ -f "$LLAMA_CPP_BUILD/src/libllama.so" ]] &&
        [[ -f "$LLAMA_CPP_BUILD/tools/mtmd/libmtmd.so" ]] &&
        [[ -f "$LLAMA_CPP_BUILD/ggml/src/libggml-cuda.so" ]]
}
