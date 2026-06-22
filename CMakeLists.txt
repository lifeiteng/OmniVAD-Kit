cmake_minimum_required(VERSION 3.15)
project(omnivad-python VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# macOS deployment floor. libomnivad.dylib gets load-time linked into consumer
# binaries (e.g. the EdgeSpeak desktop app), so its LC_BUILD_VERSION minOS must
# be <= the consumer's supported floor — otherwise dyld refuses to load it on
# older macOS ("built for newer macOS version"). With nothing set, each compiler
# bakes in the BUILD MACHINE's OS version (e.g. macOS 26), which then crashes for
# every user on an older OS. Set before FetchContent so the floor also propagates
# into the ncnn subbuild. Consumers may override with -DCMAKE_OSX_DEPLOYMENT_TARGET.
#
# FORCE is required: project() above already created an EMPTY cache entry for
# CMAKE_OSX_DEPLOYMENT_TARGET, and a plain set(... CACHE ...) is a no-op when the
# entry exists, leaving it empty. The `NOT CMAKE_OSX_DEPLOYMENT_TARGET` guard means
# we only FORCE when it is empty, so a consumer's -D (non-empty) still wins.
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS deployment version" FORCE)
endif()

# --------------------------------------------------------------------------- #
#  ncnn: build from source via FetchContent (self-contained, no system deps)   #
# --------------------------------------------------------------------------- #

include(FetchContent)

# ncnn's CMakeLists.txt uses cmake_minimum_required(VERSION 2.8.12),
# which CMake 4.x rejects. This policy override allows it to configure.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(NCNN_INSTALL OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
set(NCNN_OPENMP OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_ROTATE OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_AFFINE OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_DRAWING OFF CACHE BOOL "" FORCE)
set(NCNN_DISABLE_RTTI ON CACHE BOOL "" FORCE)
set(NCNN_DISABLE_EXCEPTION OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    ncnn
    GIT_REPOSITORY https://github.com/Tencent/ncnn.git
    GIT_TAG 20241226
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(ncnn)

# --------------------------------------------------------------------------- #
#  Build libomnivad (shared)                                                   #
# --------------------------------------------------------------------------- #

add_library(omnivad_shared SHARED
    native/src/omnivad.cpp
    native/src/chunking.cpp
    native/frontend/fft.cc
)

target_include_directories(omnivad_shared PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/native/include
    ${CMAKE_CURRENT_SOURCE_DIR}/native
)

target_compile_definitions(omnivad_shared PRIVATE OMNIVAD_SHARED)
target_link_libraries(omnivad_shared PRIVATE ncnn)
if(MSVC)
    target_compile_options(omnivad_shared PRIVATE /EHsc)
else()
    target_compile_options(omnivad_shared PRIVATE -fexceptions)
endif()

# Size optimization: LTO + strip + gc-sections
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if(IPO_SUPPORTED)
    set_target_properties(omnivad_shared PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
if(NOT MSVC)
    target_compile_options(omnivad_shared PRIVATE -ffunction-sections -fdata-sections -Os)
    if(APPLE)
        target_link_options(omnivad_shared PRIVATE -Wl,-dead_strip)
    else()
        target_link_options(omnivad_shared PRIVATE -Wl,--gc-sections -Wl,-s)
    endif()
endif()

set_target_properties(omnivad_shared PROPERTIES
    OUTPUT_NAME omnivad
)

if(APPLE)
    set_target_properties(omnivad_shared PROPERTIES
        INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    set_target_properties(omnivad_shared PROPERTIES
        INSTALL_RPATH "$ORIGIN"
    )
endif()

# --------------------------------------------------------------------------- #
#  Install into Python package                                                 #
# --------------------------------------------------------------------------- #

install(TARGETS omnivad_shared
    LIBRARY DESTINATION omnivad
    RUNTIME DESTINATION omnivad
)

install(FILES
    models/vad.omnivad
    models/stream-vad.omnivad
    models/aed.omnivad
    DESTINATION omnivad/models
)
