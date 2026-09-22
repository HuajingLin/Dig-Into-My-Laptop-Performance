# Dig into performance

Study the impact of different computer system optimization techniques 
on the performance of real-world programs.

A progressive, benchmarked demonstration of how cache-awareness, SIMD, multithreading,
and micro-kernel each contribute to raw execution speed on modern hardware,
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
n      naive     tiled     simd      threaded  micro-kernel  (GFLOP/s)
 ----------------------------------------------------------------------------------------
256    2.15      23.04     36.90     59.56     97.09       max|diff|=5.7e-06
                   10.71x     1.60x     1.61x     1.63x       (speedup over previous stage)
512    2.06      21.28     29.89     100.64    143.19      max|diff|=9.5e-06
                   10.33x     1.40x     3.37x     1.42x       (speedup over previous stage)
768    2.00      20.11     30.60     118.59    193.36      max|diff|=1.7e-05
                   10.06x     1.52x     3.87x     1.63x       (speedup over previous stage)
1024   1.13      18.48     25.40     79.54     255.86      max|diff|=1.9e-05
                   16.35x     1.37x     3.13x     3.22x       (speedup over previous stage)
1536   0.80      15.47     22.96     76.25     166.76      max|diff|=2.7e-05
                   19.34x     1.48x     3.32x     2.19x       (speedup over previous stage)
```

## Project layout

```
perf-project/
├── CMakeLists.txt
├── README.md
├── include/
│   └── matmul.h                # shared declarations
└── src/
    ├── main.cpp                # benchmark harness (timing, GFLOP/s, correctness check)
    ├── matmul_naive.cpp        # Stage 1
    ├── matmul_tiled.cpp        # Stage 2
    ├── matmul_simd.cpp         # Stage 3 
    ├── matmul_threaded.cpp     # Stage 4
    └── matmul_micro_kernel.cpp # Stage 5
    └── thread_pool.cpp         # ThreadPool implementation
```