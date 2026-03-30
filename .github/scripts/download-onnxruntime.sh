#!/usr/bin/env bash
set -euo pipefail

url="$1"
archive_path="$2"
extract_root="$3"
expected_root="$4"

if [[ -d "$expected_root" ]]; then
  echo "ONNX Runtime already present at $expected_root"
  exit 0
fi

mkdir -p "$extract_root"

if [[ ! -f "$archive_path" ]]; then
  echo "Downloading $url"
  curl -L "$url" -o "$archive_path"
else
  echo "Using cached archive $archive_path"
fi

echo "Extracting $archive_path to $extract_root"
tar -xzf "$archive_path" -C "$extract_root"

if [[ ! -d "$expected_root" ]]; then
  echo "Expected extracted directory not found: $expected_root" >&2
  exit 1
fi
