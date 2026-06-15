#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

require_linux
require_command sudo

log "installing Linux build toolchain"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    ccache \
    cmake \
    curl \
    git \
    ninja-build \
    pkg-config \
    python3 \
    python3-full \
    python3-pip \
    python3-venv \
    tar \
    unzip \
    zip \
    libopenblas-dev \
    libopencv-dev

if ! command -v rustup >/dev/null 2>&1; then
    log "installing rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
fi

if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi
rustup default stable
require_command cargo

python3 -m venv "$LINUX_VENV_DIR"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install zstandard
log "toolchain ready"
