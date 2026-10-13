#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d "Vendor/pdfium" ]; then
    echo "==> Fetching Vendor libraries (PDFium, etc.)..."
    ./scripts/fetch-vendor.sh
fi

if [ ! -d "Vendor/llama/lib" ]; then
    echo "==> Building patched llama.cpp + mtmd..."
    ./scripts/build-llama.sh
fi

echo "==> Building Prism workspace (release)..."
cargo build --workspace --release

echo "==> Prism binary built at: target/release/prism"
