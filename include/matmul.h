#pragma once
#include <cstddef>

// All matrices are square, n x n, stored row-major in a flat float array.
// C = A * B. Caller owns and zero-initializes C.

// naive matmul: textbook triple-nested loop, i-j-k order.
void matmul_naive(const float* A, const float* B, float* C, std::size_t n);