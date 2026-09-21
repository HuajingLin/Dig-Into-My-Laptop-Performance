# dig into performance

Study the impact of different computer system optimization techniques 
on the performance of real-world programs.

A progressive, benchmarked demonstration of how cache-awareness, SIMD, and
multithreading each contribute to raw execution speed on modern hardware,
using dense matrix multiplication (`C = A * B`) as the workload.

with a benchmarking harness that runs both back-to-back, 
checks their outputs agree, and reports GFLOP/s. 

# Build

Requires a C++17 compiler (GCC or Clang) and CMake ≥ 3.16.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```


## Run

```bash
./build/dig_perf                  # default sweep: 256, 512, 768, 1024, 1536
./build/dig_perf 256 1024 2048    # custom sizes
```

Sample output (11th Gen Intel i7-1185G7):
```
n      naive GFLOP/s    tiled GFLOP/s    simd GFLOP/s     threaded GFLOP/s tiled/nv  simd/tld  thrd/simd
--------------------------------------------------------------------------------------------------------
256    2.44             33.97            43.78            53.36            13.91x    1.29x     1.22x  (max|diff|=5.7e-06)
512    2.24             30.16            39.09            106.04           13.47x    1.30x     2.71x  (max|diff|=9.5e-06)
768    2.15             29.87            35.57            121.56           13.92x    1.19x     3.42x  (max|diff|=1.7e-05)
1024   1.04             26.75            34.08            135.98           25.76x    1.27x     3.99x  (max|diff|=1.9e-05)
1536   0.95             26.39            35.02            123.23           27.92x    1.33x     3.52x  (max|diff|=2.7e-05)
```

## Project layout

```
perf-project/
├── CMakeLists.txt
├── README.md
├── include/
│   └── matmul.h            # shared declarations
└── src/
    ├── main.cpp            # benchmark harness (timing, GFLOP/s, correctness check)
    ├── matmul_naive.cpp    # Stage 1
    ├── matmul_tiled.cpp    # Stage 2
    ├── matmul_simd.cpp     # Stage 3 
    └── matmul_threaded.cpp # Stage 3
```