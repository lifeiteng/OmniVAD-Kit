#!/bin/bash
# Build omnivad WASM module (simd variant by default)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMSDK="${EMSDK:-/Users/feiteng/speech/k2-fsa/emsdk}"
TOOLCHAIN="${EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"

if [ ! -f "$TOOLCHAIN" ]; then
    echo "Error: Emscripten toolchain not found at $TOOLCHAIN"
    echo "Set EMSDK env var to your emsdk root directory"
    exit 1
fi

source "${EMSDK}/emsdk_env.sh" 2>/dev/null

BUILD_DIR="${SCRIPT_DIR}/build"
OUT_DIR="${SCRIPT_DIR}/../dist/wasm"

echo "=== Building omnivad WASM (simd) ==="
cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOMNIVAD_WASM_SIMD=ON \
    -DOMNIVAD_WASM_THREADS=OFF

cmake --build "${BUILD_DIR}" -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

echo "=== Copying to ${OUT_DIR} ==="
mkdir -p "${OUT_DIR}"
cp "${BUILD_DIR}/omnivad.js" "${OUT_DIR}/"
# CJS copy for Node.js (Emscripten already includes module.exports)
cp "${BUILD_DIR}/omnivad.js" "${OUT_DIR}/omnivad.cjs"
cp "${BUILD_DIR}/omnivad.wasm" "${OUT_DIR}/"
cp "${BUILD_DIR}/omnivad.data" "${OUT_DIR}/"

echo "=== Build complete ==="
ls -lh "${OUT_DIR}"/omnivad.*
