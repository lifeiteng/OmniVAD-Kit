#!/usr/bin/env bash
#
# Memory leak detection for OmniVAD native library.
#
# On Linux:  uses valgrind (--leak-check=full)
# On macOS:  uses MallocScribble (detect use-after-free)
#
# Usage:  ./tests/test_memory.sh
#
# Prerequisites:
#   - Native library built: cmake --build native/build
#   - Model bundles in models/ directory
#   - Test audio in tests/data/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/native/build"
DATA_DIR="$SCRIPT_DIR/data"
MODELS_DIR="$PROJECT_DIR/models"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass=0
fail=0
skip=0

info()  { echo -e "${NC}$*${NC}"; }
ok()    { echo -e "${GREEN}  PASS${NC} $*"; pass=$((pass+1)); }
err()   { echo -e "${RED}  FAIL${NC} $*"; fail=$((fail+1)); }
warn()  { echo -e "${YELLOW}  SKIP${NC} $*"; skip=$((skip+1)); }

# Check build
for exe in test_nonstream_vad test_nonstream_aed test_stream_vad; do
    if [[ ! -x "$BUILD_DIR/$exe" ]]; then
        echo "ERROR: $BUILD_DIR/$exe not found. Run: cmake --build native/build"
        exit 1
    fi
done

WAV="$DATA_DIR/hello_en.wav"
WAV2="$DATA_DIR/hello_zh.wav"
[[ -f "$WAV" ]] || { echo "ERROR: $WAV not found"; exit 1; }

# Detect checker
if command -v valgrind &>/dev/null; then
    CHECKER="valgrind"
elif command -v leaks &>/dev/null; then
    CHECKER="leaks"
else
    echo "ERROR: Neither valgrind nor leaks found"
    exit 1
fi

run_check() {
    local label="$1"; shift
    local cmd=("$@")
    info ""
    info "--- $label ---"

    if [[ "$CHECKER" == "valgrind" ]]; then
        local output
        output=$(valgrind --leak-check=full --show-leak-kinds=definite \
                          --errors-for-leak-kinds=definite \
                          --undef-value-errors=no \
                          --error-exitcode=99 --log-fd=1 \
                          "${cmd[@]}" 2>/dev/null) || true
        if echo "$output" | grep -q "definitely lost: 0 bytes"; then
            ok "$label — no definite leaks"
        else
            err "$label — leaks detected"
            echo "$output" | grep -E "definitely lost|ERROR SUMMARY" | head -3
        fi
    else
        if MallocScribble=1 "${cmd[@]}" >/dev/null 2>/dev/null; then
            ok "$label — no use-after-free (MallocScribble)"
        else
            err "$label — crashed with MallocScribble"
        fi
    fi
}

echo ""
echo "================================================================"
echo "  OmniVAD Memory Leak Detection (checker: $CHECKER)"
echo "================================================================"

# Test with .omnivad bundle files
if [[ -f "$MODELS_DIR/vad.omnivad" ]]; then
    run_check "Non-stream VAD (hello_en)" \
        "$BUILD_DIR/test_nonstream_vad" "$MODELS_DIR/vad.omnivad" "$WAV" 0.4 5 200 200

    [[ -f "$WAV2" ]] && run_check "Non-stream VAD (hello_zh)" \
        "$BUILD_DIR/test_nonstream_vad" "$MODELS_DIR/vad.omnivad" "$WAV2" 0.4 5 200 200
else
    warn "Non-stream VAD — vad.omnivad not found in $MODELS_DIR"
fi

if [[ -f "$MODELS_DIR/aed.omnivad" ]]; then
    run_check "Non-stream AED (hello_en)" \
        "$BUILD_DIR/test_nonstream_aed" "$MODELS_DIR/aed.omnivad" "$WAV"

    [[ -f "$WAV2" ]] && run_check "Non-stream AED (hello_zh)" \
        "$BUILD_DIR/test_nonstream_aed" "$MODELS_DIR/aed.omnivad" "$WAV2"
else
    warn "Non-stream AED — aed.omnivad not found in $MODELS_DIR"
fi

if [[ -f "$MODELS_DIR/stream-vad.omnivad" ]]; then
    run_check "Stream VAD (hello_en)" \
        "$BUILD_DIR/test_stream_vad" "$MODELS_DIR/stream-vad.omnivad" "$WAV"

    [[ -f "$WAV2" ]] && run_check "Stream VAD (hello_zh)" \
        "$BUILD_DIR/test_stream_vad" "$MODELS_DIR/stream-vad.omnivad" "$WAV2"
else
    warn "Stream VAD — stream-vad.omnivad not found in $MODELS_DIR"
fi

echo ""
echo "================================================================"
echo -e "  Results: ${GREEN}$pass PASS${NC}, ${RED}$fail FAIL${NC}, ${YELLOW}$skip SKIP${NC}"
echo "================================================================"

[[ $fail -gt 0 ]] && exit 1 || exit 0
