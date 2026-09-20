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
n      naive GFLOP/s    tiled GFLOP/s    simd GFLOP/s     tiled/nv  simd/tld
-----------------------------------------------------------------------------
256    2.41             33.68            42.87            13.96x    1.27x  (max|diff|=5.7e-06)
512    2.23             29.77            36.79            13.35x    1.24x  (max|diff|=9.5e-06)
768    2.11             29.69            36.15            14.04x    1.22x  (max|diff|=1.7e-05)
1024   0.93             25.24            30.69            27.28x    1.22x  (max|diff|=1.9e-05)
1536   0.78             23.53            34.38            30.11x    1.46x  (max|diff|=2.7e-05)
```

## Project layout

```
perf-project/
├── CMakeLists.txt
├── README.md
├── include/
│   └── matmul.h          # shared declarations
└── src/
    ├── main.cpp           # benchmark harness (timing, GFLOP/s, correctness check)
    ├── matmul_naive.cpp   # Stage 1
    ├── matmul_tiled.cpp   # Stage 2
    └── matmul_simd.cpp    # Stage 3 
```