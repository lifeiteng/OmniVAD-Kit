cmake_minimum_required(VERSION 3.15)
project(omnivad-wasm VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------- #
#  ncnn: build from source for Emscripten                                     #
# --------------------------------------------------------------------------- #

include(FetchContent)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(NCNN_INSTALL OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
set(NCNN_OPENMP OFF CACHE BOOL "" FORCE)
set(NCNN_RUNTIME_CPU OFF CACHE BOOL "" FORCE)
set(NCNN_AVX OFF CACHE BOOL "" FORCE)
set(NCNN_AVX2 OFF CACHE BOOL "" FORCE)
set(NCNN_SIMPLEOCV ON CACHE BOOL "" FORCE)
set(NCNN_PIXEL OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_ROTATE OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_AFFINE OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_DRAWING OFF CACHE BOOL "" FORCE)
set(NCNN_DISABLE_RTTI ON CACHE BOOL "" FORCE)
set(NCNN_DISABLE_EXCEPTION ON CACHE BOOL "" FORCE)

# SIMD support (WASM SIMD maps to SSE2)
option(OMNIVAD_WASM_SIMD "Enable WASM SIMD (SSE2)" ON)
if(OMNIVAD_WASM_SIMD)
    set(NCNN_SSE2 ON CACHE BOOL "" FORCE)
else()
    set(NCNN_SSE2 OFF CACHE BOOL "" FORCE)
endif()

# Threads support
option(OMNIVAD_WASM_THREADS "Enable WASM threads" OFF)
if(OMNIVAD_WASM_THREADS)
    set(NCNN_THREADS ON CACHE BOOL "" FORCE)
    set(NCNN_SIMPLEOMP ON CACHE BOOL "" FORCE)
else()
    set(NCNN_THREADS OFF CACHE BOOL "" FORCE)
    set(NCNN_SIMPLEOMP OFF CACHE BOOL "" FORCE)
endif()

FetchContent_Declare(
    ncnn
    GIT_REPOSITORY https://github.com/Tencent/ncnn.git
    GIT_TAG 20241226
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(ncnn)

# --------------------------------------------------------------------------- #
#  omnivad WASM module                                                        #
# --------------------------------------------------------------------------- #

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

add_executable(omnivad_wasm
    ${PROJECT_ROOT}/native/src/omnivad.cpp
    ${PROJECT_ROOT}/native/src/chunking.cpp
    ${PROJECT_ROOT}/native/frontend/fft.cc
)

target_include_directories(omnivad_wasm PRIVATE
    ${PROJECT_ROOT}/native/include
    ${PROJECT_ROOT}/native
)

target_link_libraries(omnivad_wasm PRIVATE ncnn)

# Size optimization
target_compile_options(omnivad_wasm PRIVATE -Oz -fno-rtti -fno-exceptions)
set_target_properties(omnivad_wasm PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

# --------------------------------------------------------------------------- #
#  Emscripten linker flags                                                    #
# --------------------------------------------------------------------------- #

set(EXPORTED_FUNCTIONS
    "_omni_stream_vad_config_default"
    "_omni_stream_vad_create_from_buffer"
    "_omni_stream_vad_clone"
    "_omni_stream_vad_process"
    "_omni_stream_vad_process_int16"
    "_omni_stream_vad_detect_full"
    "_omni_stream_vad_detect_full_int16"
    "_omni_stream_vad_reset"
    "_omni_stream_vad_get_frame_offset"
    "_omni_stream_vad_destroy"
    "_omni_vad_create_from_buffer"
    "_omni_vad_detect"
    "_omni_vad_detect_int16"
    "_omni_vad_detect_probs"
    "_omni_vad_detect_probs_int16"
    "_omni_vad_destroy"
    "_omni_aed_create_from_buffer"
    "_omni_aed_detect"
    "_omni_aed_detect_int16"
    "_omni_aed_detect_probs"
    "_omni_aed_detect_probs_int16"
    "_omni_aed_destroy"
    "_omni_aed_overlap_segmenter_create_from_buffer"
    "_omni_aed_overlap_segmenter_clone"
    "_omni_aed_overlap_segmenter_ingest"
    "_omni_aed_overlap_segmenter_ingest_int16"
    "_omni_aed_overlap_segmenter_flush"
    "_omni_aed_overlap_segmenter_reset"
    "_omni_aed_overlap_segmenter_destroy"
    "_omni_post_config_default"
    "_omni_aed_post_config_default"
    "_omni_chunk_config_default"
    "_omni_merge_chunks"
    "_omni_error_string"
    "_omni_free"
    "_malloc"
    "_free"
)
list(JOIN EXPORTED_FUNCTIONS "," EXPORTED_FUNCTIONS_STR)

set(EMCC_FLAGS
    "-sEXPORTED_FUNCTIONS=[${EXPORTED_FUNCTIONS_STR}]"
    "-sEXPORTED_RUNTIME_METHODS=[ccall,cwrap,getValue,setValue,HEAPU8,HEAPF32,HEAP16]"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=createOmniVAD"
    "-sENVIRONMENT=web,node"
    "-sNO_EXIT_RUNTIME=1"
    "-sMALLOC=emmalloc"
    "-Oz"
    "--no-entry"
)

if(OMNIVAD_WASM_SIMD)
    list(APPEND EMCC_FLAGS "-msimd128")
endif()

if(OMNIVAD_WASM_THREADS)
    list(APPEND EMCC_FLAGS "-pthread" "-sPTHREAD_POOL_SIZE=4")
endif()

target_link_options(omnivad_wasm PRIVATE ${EMCC_FLAGS})

set_target_properties(omnivad_wasm PROPERTIES
    OUTPUT_NAME omnivad
    SUFFIX ".js"
)
