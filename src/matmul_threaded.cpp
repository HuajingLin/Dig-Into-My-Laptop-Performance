#include "matmul.h"

#include <algorithm>
#include <thread>
#include <vector>

/*
   Parallelizes matmul_simd_AVX512_block across a fixed set of threads. Each thread
   gets a contiguous, disjoint range of output rows [row_begin,row_end] and 
   runs the exact same tiled + SIMD kernel from Stage 3 on
   just that range. Because every thread only ever reads A/B and writes
   its own slice of C, there's no shared mutable state and therefore no
   locking needed.
*/
void matmul_threaded(const float* A, const float* B, float* C, std::size_t n,
                      std::size_t tile_size, unsigned num_threads) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4; // fallback if the OS won't say
    }
    // Don't spin up more threads than there are tile-rows to hand out.
    std::size_t num_tile_rows = (n + tile_size - 1) / tile_size;
    num_threads = static_cast<unsigned>(std::min<std::size_t>(num_threads, std::max<std::size_t>(num_tile_rows, 1)));

    if (num_threads <= 1) {
        matmul_simd_AVX512_block(A, B, C, n, tile_size, 0, n);
        return;
    }

    // Split whole tile-rows as evenly as possible across threads.
    std::size_t tile_rows_per_thread = num_tile_rows / num_threads;
    std::size_t remainder = num_tile_rows % num_threads;

    std::vector<std::thread> pool;
    pool.reserve(num_threads);

    std::size_t next_tile_row = 0;
    for (unsigned t = 0; t < num_threads; ++t) {
        std::size_t this_thread_tile_rows = tile_rows_per_thread + (t < remainder ? 1 : 0);
        std::size_t i_begin = next_tile_row * tile_size;
        std::size_t i_end = std::min((next_tile_row + this_thread_tile_rows) * tile_size, n);
        next_tile_row += this_thread_tile_rows;

        if (i_begin >= i_end) continue; // can happen if num_threads > num_tile_rows edge cases

        pool.emplace_back([=]() {
            matmul_simd_AVX512_block(A, B, C, n, tile_size, i_begin, i_end);
        });
    }

    for (auto& th : pool) th.join();
}
