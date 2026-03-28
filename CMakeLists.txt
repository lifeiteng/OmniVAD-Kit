cmake_minimum_required(VERSION 3.15)
project(omnivad-python VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
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
    native/frontend/fft.cc
)

target_include_directories(omnivad_shared PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/native/include
    ${CMAKE_CURRENT_SOURCE_DIR}/native
)

target_compile_definitions(omnivad_shared PRIVATE OMNIVAD_SHARED)
target_link_libraries(omnivad_shared PRIVATE ncnn)

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
