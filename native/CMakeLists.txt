cmake_minimum_required(VERSION 3.10)
project(omnivad VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --------------------------------------------------------------------------- #
#  Dependencies                                                                #
# --------------------------------------------------------------------------- #

# On macOS with Homebrew, help CMake find libomp
if(APPLE AND EXISTS "/opt/homebrew/opt/libomp")
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenMP QUIET)
if(TARGET OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP: found")
else()
    message(STATUS "OpenMP: not found (optional)")
endif()

find_package(Threads QUIET)

# NCNN discovery: NCNN_ROOT env var > cmake variable > default path
if(NOT DEFINED NCNN_ROOT)
    if(DEFINED ENV{NCNN_ROOT})
        set(NCNN_ROOT "$ENV{NCNN_ROOT}")
    else()
        # Platform-specific defaults
        if(APPLE)
            set(NCNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../3rd/prebuild/MacOS/ncnn" CACHE PATH "ncnn root")
        elseif(WIN32)
            set(NCNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../3rd/prebuild/Windows/ncnn" CACHE PATH "ncnn root")
        else()
            set(NCNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../3rd/prebuild/Linux/ncnn" CACHE PATH "ncnn root")
        endif()
    endif()
endif()

# Locate ncnn library and headers
# Try find_package first (works with Homebrew, vcpkg, etc.), then manual search
find_package(ncnn QUIET)

if(ncnn_FOUND)
    message(STATUS "ncnn found via find_package")
    set(NCNN_TARGET ncnn)
else()
    # Manual search: static or shared lib in installed or build layout
    set(_SEARCH_PATHS
        "${NCNN_ROOT}/lib/libncnn.a"
        "${NCNN_ROOT}/lib/libncnn.dylib"
        "${NCNN_ROOT}/lib/libncnn.so"
        "${NCNN_ROOT}/build/src/libncnn.a"
    )
    if(WIN32)
        list(APPEND _SEARCH_PATHS
            "${NCNN_ROOT}/lib/ncnn.lib"
            "${NCNN_ROOT}/build/src/ncnn.lib"
        )
    endif()

    set(NCNN_LIB "")
    set(NCNN_INCLUDE_DIR "")
    foreach(_path ${_SEARCH_PATHS})
        if(EXISTS "${_path}")
            set(NCNN_LIB "${_path}")
            # Determine include dir from lib location
            get_filename_component(_lib_dir "${_path}" DIRECTORY)
            get_filename_component(_parent "${_lib_dir}" DIRECTORY)
            if(EXISTS "${_parent}/include/ncnn/net.h")
                set(NCNN_INCLUDE_DIR "${_parent}/include")
            elseif(EXISTS "${_lib_dir}/ncnn/net.h")
                set(NCNN_INCLUDE_DIR "${_lib_dir}")
            elseif(EXISTS "${NCNN_ROOT}/include/ncnn/net.h")
                set(NCNN_INCLUDE_DIR "${NCNN_ROOT}/include")
            endif()
            break()
        endif()
    endforeach()

    if(NOT NCNN_LIB)
        message(FATAL_ERROR
            "ncnn not found at NCNN_ROOT=${NCNN_ROOT}\n"
            "Install ncnn (e.g. brew install ncnn) or set NCNN_ROOT.\n"
            "You can also set the NCNN_ROOT environment variable.")
    endif()
endif()

message(STATUS "NCNN_ROOT:    ${NCNN_ROOT}")
message(STATUS "NCNN include: ${NCNN_INCLUDE_DIR}")
message(STATUS "NCNN lib:     ${NCNN_LIB}")

# --------------------------------------------------------------------------- #
#  Source files                                                                #
# --------------------------------------------------------------------------- #

set(FRONTEND_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/frontend/fft.cc
)

set(API_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/omnivad.cpp
)

# --------------------------------------------------------------------------- #
#  Shared library: libomnivad                                               #
# --------------------------------------------------------------------------- #

add_library(omnivad SHARED
    ${API_SRCS}
    ${FRONTEND_SRCS}
)

# Also build a static lib for convenience
add_library(omnivad_static STATIC
    ${API_SRCS}
    ${FRONTEND_SRCS}
)

# Shared lib needs OMNIVAD_SHARED for dllexport on Windows
target_compile_definitions(omnivad PRIVATE OMNIVAD_SHARED)

# Common include directories for both targets
foreach(_target omnivad omnivad_static)
    target_include_directories(${_target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    )
    if(NOT ncnn_FOUND AND NCNN_INCLUDE_DIR)
        target_include_directories(${_target} PUBLIC
            $<BUILD_INTERFACE:${NCNN_INCLUDE_DIR}>
            $<BUILD_INTERFACE:${NCNN_INCLUDE_DIR}/ncnn>
        )
    endif()

    if(ncnn_FOUND)
        target_link_libraries(${_target} PUBLIC ncnn)
    else()
        target_link_libraries(${_target} PUBLIC ${NCNN_LIB})
    endif()

    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(${_target} PUBLIC OpenMP::OpenMP_CXX)
    endif()

    if(ANDROID)
        target_link_libraries(${_target} PUBLIC log)
    elseif(WIN32)
        # Windows: no extra libs needed
    else()
        if(Threads_FOUND)
            target_link_libraries(${_target} PUBLIC Threads::Threads)
        endif()
        target_link_libraries(${_target} PUBLIC m)
    endif()
endforeach()

set_target_properties(omnivad_static PROPERTIES OUTPUT_NAME omnivad)

# Shared library version
set_target_properties(omnivad PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# --------------------------------------------------------------------------- #
#  Test executables                                                            #
# --------------------------------------------------------------------------- #

# Test: stream VAD
add_executable(test_stream_vad test/test_stream_vad.cpp)
target_link_libraries(test_stream_vad PRIVATE omnivad_static)
target_include_directories(test_stream_vad PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Test: non-stream VAD
add_executable(test_nonstream_vad test/test_nonstream_vad.cpp)
target_link_libraries(test_nonstream_vad PRIVATE omnivad_static)
target_include_directories(test_nonstream_vad PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Test: non-stream AED
add_executable(test_nonstream_aed test/test_nonstream_aed.cpp)
target_link_libraries(test_nonstream_aed PRIVATE omnivad_static)
target_include_directories(test_nonstream_aed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Test: all 3 APIs
add_executable(test_all test/test_all.cpp)
target_link_libraries(test_all PRIVATE omnivad_static)
target_include_directories(test_all PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Test: thread-safety regression
add_executable(test_thread_safety test/test_thread_safety.cpp)
target_link_libraries(test_thread_safety PRIVATE omnivad_static)
target_include_directories(test_thread_safety PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# --------------------------------------------------------------------------- #
#  Install                                                                     #
# --------------------------------------------------------------------------- #

include(GNUInstallDirs)

install(TARGETS omnivad omnivad_static
    EXPORT omnivad-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES include/omnivad.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/omnivad
)

install(TARGETS test_stream_vad test_nonstream_vad test_nonstream_aed test_all test_thread_safety
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT omnivad-targets
    FILE omnivadTargets.cmake
    NAMESPACE omnivad::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/omnivad
)

# --------------------------------------------------------------------------- #
#  Summary                                                                     #
# --------------------------------------------------------------------------- #

message(STATUS "")
message(STATUS "=== OmniVAD Unified Library ===")
message(STATUS "Version:    ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Targets:    libomnivad (shared + static)")
message(STATUS "Tests:      test_stream_vad, test_nonstream_vad, test_nonstream_aed, test_all, test_thread_safety")
message(STATUS "")
message(STATUS "Build:   cmake --build .")
message(STATUS "Install: cmake --install . --prefix <dir>")
message(STATUS "")
