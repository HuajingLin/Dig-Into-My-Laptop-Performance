#include "matmul.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

// 64-byte aligned allocation
float* aligned_alloc_floats(std::size_t n) {
    void* ptr = nullptr;
    std::size_t bytes = n * sizeof(float);
    // round up to a multiple of the alignment, as required by std::aligned_alloc
    std::size_t alloc_bytes = (bytes + 63) & ~std::size_t(63);
#if defined(_MSC_VER)
    ptr = _aligned_malloc(alloc_bytes, 64);
#else
    ptr = std::aligned_alloc(64, alloc_bytes);
#endif
    return static_cast<float*>(ptr);
}

void aligned_free_floats(float* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

struct Matrix {
    explicit Matrix(std::size_t n_) : n(n_), data(aligned_alloc_floats(n_ * n_)) {}
    ~Matrix() { aligned_free_floats(data); }
    Matrix(const Matrix&) = delete;             //This disables copying. void duplicating the pointer.
    Matrix& operator=(const Matrix&) = delete;  //delete operator =

    std::size_t n;
    float* data;
};

//fills a Matrix with random floating-point values.
void fill_random(Matrix& m, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < m.n * m.n; ++i) m.data[i] = dist(rng);
}

//reset 0 all elements of a Matrix to zero.
void zero(Matrix& m) { 
    for (std::size_t i = 0; i < m.n * m.n; ++i) m.data[i] = 0.0f;
}

/* 
Returns seconds for the fastest of `repeats` runs (fastest-of-N is the
 standard way to filter out OS jitter/scheduling noise in microbenchmarks).
*/
template <typename Fn>  //It accepts any callable object, without incurring unnecessary copies.
double time_best_of(Fn&& fn, int repeats) {
    double best = 1e18;     //Initialize `best` to a very large number.
    for (int r = 0; r < repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();    //The highest-precision clock in the standard library
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        if (secs < best) best = secs;
    }
    return best;
}

/*
This function calculates the GFLOPS (Giga-Floating-point Operations Per Second) 
performance metric for matrix multiplication, which is the standard way 
to evaluate the efficiency of a matrix multiplication implementation.
*/
double gflops(std::size_t n, double seconds) {
    double flops = 2.0 * static_cast<double>(n) * n * n;    // one mul + one add per Multiply-Accumulate
                                                            // convert the first `n` to `double`, and the subsequent ones will be converted automatically.
    return flops / seconds / 1e9;
}

}// namespace


int main(int argc, char** argv) {
    std::vector<std::size_t> sizes = {128, 256, 512, 768, 1024};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) sizes.push_back(static_cast<std::size_t>(std::stoul(argv[i])));
    }

    std::mt19937 rng(17);

    std::printf("%-8s %-18s \n",
                "n", "naive (s / GFLOP/s)");
    std::printf("--------------------------------------------------------------------------\n");

    for (std::size_t n : sizes) {
        Matrix A(n), B(n), C_naive(n), C_tiled(n);
        fill_random(A, rng);
        fill_random(B, rng);
        zero(C_naive);
        zero(C_tiled);

        // Fewer repeats for big matrices so the sweep doesn't take forever.
        int repeats = (n <= 256) ? 5 : (n <= 512 ? 3 : 2);

        double t_naive = time_best_of(
            //A callable object that takes no arguments.
            [&]() { zero(C_naive); matmul_naive(A.data, B.data, C_naive.data, n); }, repeats);

        std::printf("%-8zu %6.3f / %-9.2f    \n",
                    n, t_naive, gflops(n, t_naive));
    }

}