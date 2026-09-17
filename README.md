# dig-perf

A progressive, benchmarked demonstration of how cache-awareness, SIMD, and
multithreading each contribute to raw execution speed on modern hardware,
using dense matrix multiplication (`C = A * B`) as the workload.

This repo currently implements **Stage 1** (naive baseline) and **Stage 2**
(cache-tiled + loop-reordered), with a benchmarking harness that runs both
back-to-back, checks their outputs agree, and reports GFLOP/s. 

# Build

Requires a C++17 compiler (GCC or Clang) and CMake ≥ 3.16.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```


## Run

```bash
./build/dig_perf                  # default sweep: 128, 256, 512, 768, 1024
./build/dig_perf 256 1024 2048    # custom sizes
```

Sample output (11th Gen Intel i7-1185G7):
```
n        naive (s / GFLOP/s) tiled (s / GFLOP/s) speedup    max|diff|
--------------------------------------------------------------------------
128       0.002 / 1.74       0.000 / 26.95      15.47x    2.86e-06
256       0.014 / 2.46       0.001 / 27.43      11.16x    7.63e-06
512       0.118 / 2.28       0.011 / 25.40      11.14x    1.14e-05
768       0.420 / 2.16       0.034 / 26.51      12.29x    1.62e-05
1024      2.014 / 1.07       0.093 / 23.15      21.71x    2.67e-05
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
    └── matmul_tiled.cpp   # Stage 2
```