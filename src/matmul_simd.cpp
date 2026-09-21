#include "matmul.h"
#include <algorithm>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define PERF_MATMUL_HAVE_AVX2_FMA 1
#else
#define PERF_MATMUL_HAVE_AVX2_FMA 0
#endif

/* 
Same tile/loop structure as matmul_tiled, but the innermost j loop is hand-vectorized: 
instead of one float per iteration we process 8 floats at a time with a fused multiply-add 
AVX2 instruction (c[j..j+7] += a_ik * b[j..j+7] in a single instruction, versus 8 separate 
scalar mul+add pairs). This is the same transformation a good auto-vectorizer performs, 
made explicit so it's visible and guaranteed regardless of compiler heuristics.

Unaligned loads/stores (_mm256_loadu_ps/storeu_ps) are used throughout for simplicity and 
correctness on any tile_size/n combination; on this generation of Intel hardware unaligned 
AVX2 loads on cache-line-aligned data cost essentially nothing extra.
*/
void matmul_simd__AVX2_block(const float* A, const float* B, float* C, std::size_t n,
                        std::size_t tile_size, std::size_t i_begin, std::size_t i_end) {
    constexpr std::size_t kLanes = 8; // 8 x float32 per AVX2 register

    for (std::size_t ii = i_begin; ii < i_end; ii += tile_size) {
        std::size_t i_max = std::min(ii + tile_size, i_end);
        for (std::size_t kk = 0; kk < n; kk += tile_size) {
            std::size_t k_max = std::min(kk + tile_size, n);
            for (std::size_t jj = 0; jj < n; jj += tile_size) {
                std::size_t j_max = std::min(jj + tile_size, n);
                std::size_t j_vec_end = jj + ((j_max - jj) / kLanes) * kLanes;

                for (std::size_t i = ii; i < i_max; ++i) {
                    for (std::size_t k = kk; k < k_max; ++k) {
                        float a_ik = A[i * n + k];
                        const float* b_row = &B[k * n];
                        float* c_row = &C[i * n];

#if PERF_MATMUL_HAVE_AVX2_FMA
                        __m256 a_vec = _mm256_set1_ps(a_ik);    //broadcast a float to 8 units
                        std::size_t j = jj;
                        for (; j < j_vec_end; j += kLanes) {    //One float is 32 bit, 256 / 32 = 8
                            __m256 b_vec = _mm256_loadu_ps(&b_row[j]);  //load a float for 8 times
                            __m256 c_vec = _mm256_loadu_ps(&c_row[j]);  // ~
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);   //core: Fused Multiply-Add completed 8 floats a time.
                            _mm256_storeu_ps(&c_row[j], c_vec);         // store to C
                        }
                        // scalar cleanup for the tail (j_max - jj) % 8 elements
                        for (; j < j_max; ++j) {
                            c_row[j] += a_ik * b_row[j];
                        }
#else
                        // Portable fallback when AVX2+FMA isn't available
                        // at compile time (e.g. building without
                        // -march=native, or on non-x86 hardware).
                        for (std::size_t j = jj; j < j_max; ++j) {
                            c_row[j] += a_ik * b_row[j];
                        }
#endif
                    }
                }
            }
        }
    }
}


void matmul_simd_AVX512_block(const float* A, const float* B, float* C, std::size_t n,
                        std::size_t tile_size, std::size_t i_begin, std::size_t i_end) {
    constexpr std::size_t kLanes = 16; // 16 x float32 per AVX512 register

    for (std::size_t ii = i_begin; ii < i_end; ii += tile_size) {
        std::size_t i_max = std::min(ii + tile_size, i_end);
        for (std::size_t kk = 0; kk < n; kk += tile_size) {
            std::size_t k_max = std::min(kk + tile_size, n);
            for (std::size_t jj = 0; jj < n; jj += tile_size) {
                std::size_t j_max = std::min(jj + tile_size, n);
                std::size_t j_vec_end = jj + ((j_max - jj) / kLanes) * kLanes;

                for (std::size_t i = ii; i < i_max; ++i) {
                    for (std::size_t k = kk; k < k_max; ++k) {
                        float a_ik = A[i * n + k];
                        const float* b_row = &B[k * n];
                        float* c_row = &C[i * n];

#if PERF_MATMUL_HAVE_AVX2_FMA
                        __m512  a_vec = _mm512_set1_ps(a_ik);    //broadcast a float to 16 units
                        std::size_t j = jj;
                        for (; j < j_vec_end; j += kLanes) {    //One float is 32 bit, 512 / 32 = 16
                            __m512  b_vec = _mm512_loadu_ps(&b_row[j]);  //load a float for 16 times
                            __m512  c_vec = _mm512_loadu_ps(&c_row[j]);  // ~
                            c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);   //core: Fused Multiply-Add completed 8 floats a time.
                            _mm512_storeu_ps(&c_row[j], c_vec);         // store to C
                        }
                        // scalar cleanup for the tail (j_max - jj) % 8 elements
                        for (; j < j_max; ++j) {
                            c_row[j] += a_ik * b_row[j];
                        }
#else
                        // Portable fallback when AVX2+FMA isn't available
                        // at compile time (e.g. building without
                        // -march=native, or on non-x86 hardware).
                        for (std::size_t j = jj; j < j_max; ++j) {
                            c_row[j] += a_ik * b_row[j];
                        }
#endif
                    }
                }
            }
        }
    }
}

void matmul_simd(const float* A, const float* B, float* C, std::size_t n,
                  std::size_t tile_size) {
    matmul_simd_AVX512_block(A, B, C, n, tile_size, 0, n);
}
