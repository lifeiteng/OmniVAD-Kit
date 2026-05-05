#!/bin/bash
# Build omnivad WASM module (simd variant by default).
#
# Requires the EMSDK env var to point at your emsdk root:
#   EMSDK=/path/to/emsdk packages/omnivad/wasm/build.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${EMSDK}" ]; then
    echo "Error: EMSDK env var is not set." >&2
    echo "       Set it to your emsdk root, e.g.:" >&2
    echo "         EMSDK=/path/to/emsdk $0" >&2
    exit 1
fi

TOOLCHAIN="${EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"
if [ ! -f "$TOOLCHAIN" ]; then
    echo "Error: Emscripten toolchain not found at $TOOLCHAIN" >&2
    echo "       Check that EMSDK ($EMSDK) points at a real emsdk install." >&2
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
cp "${BUILD_DIR}/omnivad.js"   "${OUT_DIR}/"
# CJS copy for Node.js (Emscripten already includes module.exports)
cp "${BUILD_DIR}/omnivad.js"   "${OUT_DIR}/omnivad.cjs"
cp "${BUILD_DIR}/omnivad.wasm" "${OUT_DIR}/"
# omnivad.data is only generated when --preload-file is used. Our build
# embeds models on the JS side, so this file is normally absent — copy
# it best-effort to stay forward-compatible.
if [ -f "${BUILD_DIR}/omnivad.data" ]; then
    cp "${BUILD_DIR}/omnivad.data" "${OUT_DIR}/"
fi

echo "=== Build complete ==="
ls -lh "${OUT_DIR}"/omnivad.*
