#include "matmul.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

/* 
64-byte aligned allocation, so rows/tiles line up cleanly with cache
 lines and (later, when I add SIMD) with AVX2/AVX-512 load widths.
 */
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

//computes the maximum absolute element-wise difference between two matrices
double max_abs_diff(const Matrix& a, const Matrix& b) { 
    double worst = 0.0;
    for (std::size_t i = 0; i < a.n * a.n; ++i) {
        double d = std::abs(static_cast<double>(a.data[i]) - static_cast<double>(b.data[i]));
        if (d > worst) worst = d;
    }
    return worst;
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

//for comparing the performance between my own imolementations and OpenBLAS.
struct Stage {
    std::string name;
    std::unique_ptr<Matrix> result;      // this stage's own C buffer
    std::function<void(float*)> compute; // fills the given C buffer for one n x n run
};

}// namespace


int main(int argc, char** argv) {
    std::vector<std::size_t> sizes = {256, 512, 768, 1024, 1536, 2048};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) sizes.push_back(static_cast<std::size_t>(std::stoul(argv[i])));
    }

    std::mt19937 rng(17);

    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 4; // fallback if the OS won't say
    std::printf("hardware_concurrency() reports %u threads\n\n", hw_threads);

    //pool is created ONCE, outside the timed benchmark loop, and reused for every matrix size.
    ThreadPool pool(hw_threads);
    constexpr std::size_t tile_size = 128;

#ifdef TEMPERLORLY_CANCEL_FOR_COMPARISON
    std::printf("%-6s %-9s %-9s %-9s %-9s %-9s   (GFLOP/s)\n",
                "n", "naive", "tiled", "simd", "threaded", "micro-kernel");
    std::printf("-------------------------------------------------------------------------------------------\n");
    
    for (std::size_t n : sizes) {
        Matrix A(n), B(n), C_naive(n), C_tiled(n), C_simd(n), C_threaded(n), C_micro_kernel(n);
        fill_random(A, rng);
        fill_random(B, rng);

        // Fewer repeats for big matrices so the sweep doesn't take forever.
        int repeats = (n <= 256) ? 5 : (n <= 512 ? 3 : 2);
        
        double t_naive = time_best_of(
            [&]() { zero(C_naive); matmul_naive(A.data, B.data, C_naive.data, n); }, repeats);

        double t_tiled = time_best_of(
            [&]() { zero(C_tiled); matmul_tiled(A.data, B.data, C_tiled.data, n, tile_size); },  //A callable object that takes no arguments.
            repeats);

        double t_simd = time_best_of(
            [&]() { zero(C_simd); matmul_simd(A.data, B.data, C_simd.data, n, tile_size); }, repeats);

         double t_threaded = time_best_of(
            [&]() { zero(C_threaded); matmul_threaded(A.data, B.data, C_threaded.data, n, tile_size, /*num_threads=*/0); },
            repeats);

        double t_micro_kernel = time_best_of(
            [&]() { zero(C_micro_kernel); matmul_micro_kernel(A.data, B.data, C_micro_kernel.data, n, pool, tile_size); },
            repeats);
        
        /*
        Sanity check: every stage should agree with the naive baseline to 
        within floating-point rounding error.
        */
        double diff_tiled = max_abs_diff(C_naive, C_tiled);
        double diff_simd = max_abs_diff(C_naive, C_simd);
        double diff_threaded = max_abs_diff(C_naive, C_threaded);
        double diff_micro_kernel = max_abs_diff(C_naive, C_micro_kernel);
        double worst_diff = std::max({diff_tiled, diff_simd, diff_threaded, diff_micro_kernel});

        std::printf("%-6zu %-9.2f %-9.2f %-9.2f %-9.2f %-9.2f   max|diff|=%.1e%s\n",
                    n, gflops(n, t_naive), gflops(n, t_tiled), gflops(n, t_simd),
                    gflops(n, t_threaded), gflops(n, t_micro_kernel),
                    worst_diff, worst_diff > 1e-2 ? "  <-- CHECK THIS" : "");
        std::printf("%-6s %8s %8.2fx %8.2fx %8.2fx %8.2fx       (speedup over previous stage)\n",
                    "", "", t_naive / t_tiled, t_tiled / t_simd, t_simd / t_threaded, t_threaded / t_micro_kernel);

    }
#endif    
    //for comparing the performance between my own imolementations and OpenBLAS.
    for (std::size_t n : sizes) {
        Matrix A(n), B(n);
        fill_random(A, rng);
        fill_random(B, rng);

        std::vector<Stage> stages;
        auto add_stage = [&](const std::string& name, std::function<void(float*)> compute) {
            stages.push_back(Stage{name, std::make_unique<Matrix>(n), std::move(compute)});
        };

        add_stage("my_matmul", [&](float* c) { matmul_micro_kernel(A.data, B.data, c, n, pool, tile_size); });
#ifdef PERF_MATMUL_HAVE_OPENBLAS
        add_stage("OpenBLAS", [&](float* c) {
            matmul_openblas_set_threads(static_cast<int>(hw_threads));
            matmul_openblas(A.data, B.data, c, n);
        });
#else
        std::printf("OpenBLAS not available, skipping comparison.\n");
        continue;
#endif
        // Fewer repeats for big matrices so the sweep doesn't take forever.
        int repeats = (n <= 256) ? 5 : (n <= 512 ? 3 : 2);

        std::vector<double> times(stages.size());
        for (std::size_t i = 0; i < stages.size(); ++i) {
            Matrix& result = *stages[i].result;
            const auto& compute = stages[i].compute;
            times[i] = time_best_of([&]() { zero(result); compute(result.data); }, repeats);
        }

        // Sanity check: every stage should agree with stage 0 (naive) to
        // within floating-point rounding error.
        double worst_diff = 0.0;
        for (std::size_t i = 1; i < stages.size(); ++i) {
            worst_diff = std::max(worst_diff, max_abs_diff(*stages[0].result, *stages[i].result));
        }

        if (n == sizes.front()) {   // print header only once
            std::cout << std::left << std::setw(6) << "n";
            for (auto& s : stages) std::cout << std::setw(10) << s.name;
            std::cout << "  (GFLOP/s)\n";
            std::cout << std::string(6 + 10 * stages.size() + 12, '-') << "\n";
        }

        std::cout << std::left << std::setw(6) << n;
        for (std::size_t i = 0; i < stages.size(); ++i) {   //print the GFLOPS for each stage
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << gflops(n, times[i]);
        }
        std::cout << "  max|diff|=" << std::scientific << std::setprecision(1) << worst_diff;
        if (worst_diff > 1e-2) std::cout << "  <-- CHECK THIS";
        std::cout << "\n";

        std::cout << std::string(6, ' ');
        std::cout << std::setw(10) << " "; // naive has no "speedup over previous" entry
        for (std::size_t i = 1; i < stages.size(); ++i) {
            std::ostringstream ratio;
            ratio <<"   "<< std::fixed << std::setprecision(2) << (times[i - 1] / times[i]) << "x";
            std::cout << std::setw(12) << ratio.str();
        }
        std::cout << "  (speedup over mine)\n";

    }

    return 0;

}