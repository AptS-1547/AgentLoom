#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

require_linux
for command in cmake ninja python3; do
    require_command "$command"
done
require_command cargo
[[ -x "$PYTHON_BIN" ]] || fail "Python venv not found at $PYTHON_BIN; run linux/scripts/bootstrap_toolchain.sh first"

enable_inference=false
if [[ "${1:-}" == "--inference" ]]; then
    enable_inference=true
    has_cuda_llama_binary || fail "--inference requires a prepared CUDA llama.cpp binary; set LLAMA_CPP_CUDA_URL and run prepare_deps.sh"
fi

shim_args=()
if [[ "$enable_inference" != true ]]; then
    shim_args+=(--disable-inference)
fi

"$PYTHON_BIN" "$SCRIPT_DIR/apply_cmake_shim.py" "$REPO_ROOT" "$LINUX_SOURCE_DIR" "${shim_args[@]}"
ln -sfn "$DEPS_DIR" "$LINUX_SOURCE_DIR/deps"
ln -sfn "$VCPKG_INSTALL_ROOT" "$LINUX_SOURCE_DIR/vcpkg_installed"

cmake -S "$LINUX_SOURCE_DIR" -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS" \
    -DBERT_BUILD_TESTS=ON \
    -DBERT_BUILD_EMOTION_INFERENCE_SERVER=ON \
    -DBERT_BUILD_MULTIMODAL_INFERENCE_SERVER="$enable_inference" \
    -DBERT_VCPKG_TRIPLET="$VCPKG_TRIPLET" \
    -DBERT_USE_ONNXRUNTIME_GPU=OFF \
    -DONNXRUNTIME_CPU_ROOT="$ONNXRUNTIME_ROOT" \
    -DBERT_SQLITE_INCLUDE_DIR="$SQLITE_ROOT" \
    -DBERT_BOOST_ROOT="$BOOST_ROOT" \
    -DBERT_EIGEN_ROOT="$EIGEN_ROOT" \
    -DBERT_FAISS_ROOT="$FAISS_ROOT" \
    -DBERT_MKL_RUNTIME_ROOT="$MKL_ROOT" \
    -DCARGO_EXECUTABLE="$(command -v cargo)" \
    -DLLAMA_CPP_ROOT="$LLAMA_CPP_ROOT" \
    -DLLAMA_CPP_BUILD="$LLAMA_CPP_BUILD" \
    -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4

log "configured Linux build at $BUILD_DIR"
