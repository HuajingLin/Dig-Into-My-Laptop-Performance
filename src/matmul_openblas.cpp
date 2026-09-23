#include "matmul.h"
#include <cblas.h>


extern "C" void openblas_set_num_threads(int num_threads);

void matmul_openblas(const float* A, const float* B, float* C, std::size_t n) {
    /* Row-major, no transpose on either operand, alpha=1, beta=0 -- this
       computes exactly C = A * B, matching every other stage in this
       project so the same correctness check (diff against matmul_naive)
       applies unchanged.*/
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<blasint>(n), static_cast<blasint>(n), static_cast<blasint>(n),
                1.0f, A, static_cast<blasint>(n),
                B, static_cast<blasint>(n),
                0.0f, C, static_cast<blasint>(n));
}

void matmul_openblas_set_threads(int num_threads) {
    openblas_set_num_threads(num_threads);
}
