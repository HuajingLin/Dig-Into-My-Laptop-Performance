#include "matmul.h"
#include <algorithm>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define PERF_MATMUL_HAVE_AVX2_FMA 1
#else
#define PERF_MATMUL_HAVE_AVX2_FMA 0
#endif

namespace {

constexpr std::size_t kMR = 8; // output rows per micro-kernel call
constexpr std::size_t kNR = 8; // output columns per micro-kernel call (= 1 AVX2 register)

#if PERF_MATMUL_HAVE_AVX2_FMA
/*Computes a 4-row x 8-column tile of C, accumulating over k in [k0, k1).  The key difference 
from Stage 3: each B vector (b_vec) is loaded ONCE per k and then reused in 4 separate FMAs, 
one per output row, via 4 independent accumulator registers. Stage 3's kernel processes one 
output row at a time, so it reloads/reuses a B vector only once before moving on -- this kernel 
gets 4x the arithmetic per B load, which is exactly what you want when memory bandwidth 
(not FMA throughput) is the bottleneck, as Stage 3's benchmark numbers showed it was.*/
inline void micro_kernel_4x8(const float* A, const float* B, float* C, std::size_t n,
                              std::size_t k0, std::size_t k1, std::size_t i0, std::size_t j0) {
    __m256 acc0 = _mm256_loadu_ps(&C[(i0 + 0) * n + j0]);
    __m256 acc1 = _mm256_loadu_ps(&C[(i0 + 1) * n + j0]);
    __m256 acc2 = _mm256_loadu_ps(&C[(i0 + 2) * n + j0]);
    __m256 acc3 = _mm256_loadu_ps(&C[(i0 + 3) * n + j0]);

    const float* a_row0 = &A[(i0 + 0) * n];
    const float* a_row1 = &A[(i0 + 1) * n];
    const float* a_row2 = &A[(i0 + 2) * n];
    const float* a_row3 = &A[(i0 + 3) * n];

    for (std::size_t k = k0; k < k1; ++k) {
        __m256 b_vec = _mm256_loadu_ps(&B[k * n + j0]); // 1 load, reused 4x below
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a_row0[k]), b_vec, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a_row1[k]), b_vec, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a_row2[k]), b_vec, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a_row3[k]), b_vec, acc3);
    }

    _mm256_storeu_ps(&C[(i0 + 0) * n + j0], acc0);
    _mm256_storeu_ps(&C[(i0 + 1) * n + j0], acc1);
    _mm256_storeu_ps(&C[(i0 + 2) * n + j0], acc2);
    _mm256_storeu_ps(&C[(i0 + 3) * n + j0], acc3);
}

inline void micro_kernel_8x8(
    const float* A,
    const float* B,
    float* C,
    std::size_t n,
    std::size_t k0,
    std::size_t k1,
    std::size_t i0,
    std::size_t j0)
{
    // 8 x 8 accumulators
    __m256 acc0 = _mm256_loadu_ps(&C[(i0 + 0) * n + j0]);
    __m256 acc1 = _mm256_loadu_ps(&C[(i0 + 1) * n + j0]);
    __m256 acc2 = _mm256_loadu_ps(&C[(i0 + 2) * n + j0]);
    __m256 acc3 = _mm256_loadu_ps(&C[(i0 + 3) * n + j0]);
    __m256 acc4 = _mm256_loadu_ps(&C[(i0 + 4) * n + j0]);
    __m256 acc5 = _mm256_loadu_ps(&C[(i0 + 5) * n + j0]);
    __m256 acc6 = _mm256_loadu_ps(&C[(i0 + 6) * n + j0]);
    __m256 acc7 = _mm256_loadu_ps(&C[(i0 + 7) * n + j0]);

    const float* a_row0 = &A[(i0 + 0) * n];
    const float* a_row1 = &A[(i0 + 1) * n];
    const float* a_row2 = &A[(i0 + 2) * n];
    const float* a_row3 = &A[(i0 + 3) * n];
    const float* a_row4 = &A[(i0 + 4) * n];
    const float* a_row5 = &A[(i0 + 5) * n];
    const float* a_row6 = &A[(i0 + 6) * n];
    const float* a_row7 = &A[(i0 + 7) * n];

    for (std::size_t k = k0; k < k1; ++k) {
        // 1 B load -> reused by all 8 rows
        __m256 b_vec = _mm256_loadu_ps(&B[k * n + j0]);

        acc0 = _mm256_fmadd_ps(
            _mm256_set1_ps(a_row0[k]), b_vec, acc0);

        acc1 = _mm256_fmadd_ps(
            _mm256_set1_ps(a_row1[k]), b_vec, acc1);

        acc2 = _mm256_fmadd_ps(
            _mm256_set1_ps(a_row2[k]), b_vec, acc2);

        acc3 = _mm256_fmadd_ps(
            _mm256_set1_ps(a_row3[k]), b_vec, acc3);

        acc4 = _mm256_fmadd_ps(
            _mm256_set1_ps(a_row4[k]), b_vec, acc4);

        acc5 = _mm256_fmadd_ps(
            _mm256_set1_ps(a_row5[k]), b_vec, acc5);

        acc6 = _mm256_fmadd_ps(
            _mm256_set1_ps(a_row6[k]), b_vec, acc6);

        acc7 = _mm256_fmadd_ps(
            _mm256_set1_ps(a_row7[k]), b_vec, acc7);
    }
    //write back to C.
    _mm256_storeu_ps(&C[(i0 + 0) * n + j0], acc0);
    _mm256_storeu_ps(&C[(i0 + 1) * n + j0], acc1);
    _mm256_storeu_ps(&C[(i0 + 2) * n + j0], acc2);
    _mm256_storeu_ps(&C[(i0 + 3) * n + j0], acc3);
    _mm256_storeu_ps(&C[(i0 + 4) * n + j0], acc4);
    _mm256_storeu_ps(&C[(i0 + 5) * n + j0], acc5);
    _mm256_storeu_ps(&C[(i0 + 6) * n + j0], acc6);
    _mm256_storeu_ps(&C[(i0 + 7) * n + j0], acc7);
}

/*Single-row fallback for the edges the 4x8 micro-kernel can't cover 
(row or column counts not divisible by kMR/kNR -- always possible at 
the last tile when n isn't a multiple of tile_size). Same vectorized 
single-row approach as Stage 3, restricted to columns [j0, j1).*/
inline void accumulate_row_range(const float* A, const float* B, float* C, std::size_t n,
                                  std::size_t i, std::size_t k0, std::size_t k1,
                                  std::size_t j0, std::size_t j1) {
    constexpr std::size_t kLanes = 8;
    std::size_t j_vec_end = j0 + ((j1 - j0) / kLanes) * kLanes;

    for (std::size_t k = k0; k < k1; ++k) {
        float a_ik = A[i * n + k];
        __m256 a_vec = _mm256_set1_ps(a_ik);
        const float* b_row = &B[k * n];
        float* c_row = &C[i * n];

        std::size_t j = j0;
        for (; j < j_vec_end; j += kLanes) {
            __m256 b_vec = _mm256_loadu_ps(&b_row[j]);
            __m256 c_vec = _mm256_loadu_ps(&c_row[j]);
            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            _mm256_storeu_ps(&c_row[j], c_vec);
        }
        for (; j < j1; ++j) c_row[j] += a_ik * b_row[j];
    }
}
#endif // PERF_MATMUL_HAVE_AVX2_FMA

} // namespace

/*Same cache-tiling structure as Stages 2-4 (ii/kk/jj sweeping tile_size blocks), 
but within each tile the register-blocked micro-kernel handles as much of the tile as divides evenly 
into 4x8 chunks, with two small edge loops mopping up any leftover rows/columns at tile boundaries.*/
void matmul_register_blocked_block(const float* A, const float* B, float* C, std::size_t n,
                                    std::size_t tile_size, std::size_t i_begin, std::size_t i_end) {
#if !PERF_MATMUL_HAVE_AVX2_FMA
    /*No AVX2/FMA at compile time: register blocking without a wide accumulator doesn't buy anything, 
    so just reuse Stage 3's kernel (which itself falls back to portable scalar code in this case).*/
    matmul_simd_AVX512_block(A, B, C, n, tile_size, i_begin, i_end);
#else
    for (std::size_t ii = i_begin; ii < i_end; ii += tile_size) {
        std::size_t i_max = std::min(ii + tile_size, i_end);
        for (std::size_t kk = 0; kk < n; kk += tile_size) {
            std::size_t k_max = std::min(kk + tile_size, n);
            for (std::size_t jj = 0; jj < n; jj += tile_size) {
                std::size_t j_max = std::min(jj + tile_size, n);

                std::size_t i_reg_end = ii + ((i_max - ii) / kMR) * kMR;
                std::size_t j_reg_end = jj + ((j_max - jj) / kNR) * kNR;

                // Main register-blocked region: full 4x8 micro-tiles.
                for (std::size_t i = ii; i < i_reg_end; i += kMR) {
                    for (std::size_t j = jj; j < j_reg_end; j += kNR) {
                        micro_kernel_8x8(A, B, C, n, kk, k_max, i, j);
                    }
                    /* Column tail for this row-group: columns that 
                    didn't divide evenly into an 8-wide micro-tile.*/
                    if (j_reg_end < j_max) {
                        for (std::size_t r = i; r < i + kMR; ++r) {
                            accumulate_row_range(A, B, C, n, r, kk, k_max, j_reg_end, j_max);
                        }
                    }
                }
                /*Row tail: rows that didn't divide evenly into a 4-row group, 
                handled one row at a time across the full column range of this tile.*/
                for (std::size_t i = i_reg_end; i < i_max; ++i) {
                    accumulate_row_range(A, B, C, n, i, kk, k_max, jj, j_max);
                }
            }
        }
    }
#endif
}

// Partitions rows into contiguous, tile-aligned chunks across the pool's worker threads.
void matmul_micro_kernel(const float* A, const float* B, float* C, std::size_t n,
                    ThreadPool& pool, std::size_t tile_size) {
    std::size_t num_tile_rows = (n + tile_size - 1) / tile_size;

    pool.run([&](unsigned thread_index, unsigned num_threads) {
        std::size_t usable_threads = std::min<std::size_t>(num_threads, std::max<std::size_t>(num_tile_rows, 1));
        if (thread_index >= usable_threads) return; // more worker threads than tile-rows to hand out

        std::size_t tile_rows_per_thread = num_tile_rows / usable_threads;
        std::size_t remainder = num_tile_rows % usable_threads;
        std::size_t this_thread_tile_rows = tile_rows_per_thread + (static_cast<std::size_t>(thread_index) < remainder ? 1 : 0);
        std::size_t tile_start = static_cast<std::size_t>(thread_index) * tile_rows_per_thread +
                                  std::min<std::size_t>(thread_index, remainder);

        std::size_t i_begin = tile_start * tile_size;
        std::size_t i_end = std::min((tile_start + this_thread_tile_rows) * tile_size, n);
        if (i_begin >= i_end) return;

        matmul_register_blocked_block(A, B, C, n, tile_size, i_begin, i_end);
    });
}
