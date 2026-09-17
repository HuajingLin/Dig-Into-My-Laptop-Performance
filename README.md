# dig-perf

This repo currently implements naive matmul for baseline.

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
n        naive (s / GFLOP/s)
-------------------------------
128         0.002 / 2.60
256         0.014 / 2.42
512         0.118 / 2.27
768         0.424 / 2.14
1024        2.042 / 1.05
```