#pragma once
#include <cstddef>

// All matrices are square, n x n, stored row-major in a flat float array.
// C = A * B. Caller owns and zero-initializes C.

// Stage 1: naive matmul, textbook triple-nested loop, i-j-k order.
void matmul_naive(const float* A, const float* B, float* C, std::size_t n);

// Stage 2: cache-blocked (tiled) + loop-reordered (i-k-j).
void matmul_tiled(const float* A, const float* B, float* C, std::size_t n,
                   std::size_t tile_size = 128);

// Stage 3:  the innermost j loop is hand-vectorized with AVX2 FMA intrinsics.
void matmul_simd(const float* A, const float* B, float* C, std::size_t n,
                  std::size_t tile_size = 128);
