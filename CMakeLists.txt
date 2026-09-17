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
elseif(MSVC)
    add_compile_options(/O2 /arch:AVX2)
endif()

add_executable(dig_perf
    src/main.cpp
    src/matmul_naive.cpp
    src/matmul_tiled.cpp
)

target_include_directories(dig_perf PRIVATE include)

find_package(Threads REQUIRED)
target_link_libraries(dig_perf PRIVATE Threads::Threads)