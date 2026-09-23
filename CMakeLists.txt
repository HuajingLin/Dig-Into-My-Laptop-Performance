cmake_minimum_required(VERSION 3.16)
project(dig_perf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# optimization options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra)
    add_compile_options(-fopt-info-vec-optimized)
elseif(MSVC)
    add_compile_options(/O2 /arch:AVX2)
endif()

add_executable(dig_perf
    src/main.cpp
    src/matmul_naive.cpp
    src/matmul_tiled.cpp
    src/matmul_simd.cpp
    src/matmul_threaded.cpp
    src/matmul_micro_kernel.cpp
    src/thread_pool.cpp
)

target_include_directories(dig_perf PRIVATE include)

find_package(Threads REQUIRED)
target_link_libraries(dig_perf PRIVATE Threads::Threads)

# --------  OpenBLAS comparison -------
find_package(BLAS QUIET)
find_path(PERF_MATMUL_CBLAS_INCLUDE_DIR cblas.h
    PATHS /usr/include /usr/include/openblas /usr/local/include
          /usr/local/opt/openblas/include /opt/homebrew/opt/openblas/include
)

if(BLAS_FOUND AND PERF_MATMUL_CBLAS_INCLUDE_DIR)
    message(STATUS "dig_perf: BLAS found (${BLAS_LIBRARIES}) -- comparison OpenBLAS enabled.")
    message(STATUS "dig_perf: NOTE -- find_package(BLAS) picks up whatever BLAS provider")
    message(STATUS "  is configured as the system default. Confirm the library above is")
    message(STATUS "  actually OpenBLAS (not reference BLAS/ATLAS) if the comparison looks")
    message(STATUS "  off -- e.g. `ldd build/dig_perf | grep -i blas` after building.")
    target_sources(dig_perf PRIVATE src/matmul_openblas.cpp)
    target_compile_definitions(dig_perf PRIVATE PERF_MATMUL_HAVE_OPENBLAS)
    target_include_directories(dig_perf PRIVATE ${PERF_MATMUL_CBLAS_INCLUDE_DIR})
    target_link_libraries(dig_perf PRIVATE ${BLAS_LIBRARIES})
else()
    message(STATUS "dig_perf: BLAS/cblas.h not found -- OpenBLAS comparison")
    message(STATUS "  will be skipped. Install libopenblas-dev (or equivalent) and")
    message(STATUS "  reconfigure to enable it.")
endif()