#include "matmul.h"
#include <algorithm>
/*
Reordering to i-k-j makes the innermost loop stream through B row-wise
(unit stride), and tiling keeps the working set of A/B/C blocks resident
in L1/L2 cache instead of thrashing on every pass over a large matrix.
*/
void matmul_tiled(const float* A, const float* B, float* C, std::size_t n,
                   std::size_t tile_size) {
    for (std::size_t ii = 0; ii < n; ii += tile_size) {     //Cut into block
        std::size_t i_max = std::min(ii + tile_size, n);    //The end boundary of the tile. This handles the case where n is not divisible by tile_size.
        for (std::size_t kk = 0; kk < n; kk += tile_size) {
            std::size_t k_max = std::min(kk + tile_size, n);
            for (std::size_t jj = 0; jj < n; jj += tile_size) {
                std::size_t j_max = std::min(jj + tile_size, n);

                for (std::size_t i = ii; i < i_max; ++i) {  //computation within the block
                    for (std::size_t k = kk; k < k_max; ++k) {
                        float a_ik = A[i * n + k];
                        const float* b_row = &B[k * n];     //The address of B is calculated in advance.
                        float* c_row = &C[i * n];
                        for (std::size_t j = jj; j < j_max; ++j) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}
